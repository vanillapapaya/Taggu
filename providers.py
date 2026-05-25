"""VLM Provider 추상화 — 로컬(Qwen) / OpenAI / Anthropic / Gemini.

각 provider는 (이미지 경로) → (한국어 태그 CSV, 한국어 설명) 반환.
이미지 분석은 동기 호출 (전체 backfill 루프가 별도 스레드에서 돔).
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Optional, Protocol


# 모든 provider 공통 프롬프트 — 일관된 결과 위해 동일하게 사용
VLM_PROMPT = """이 이미지를 분석해서 다음 형식으로 응답해줘:

태그: (쉼표로 구분된 한국어 태그 5-10개. 캐릭터명, 작품명, 분위기, 상황 등)
설명: (한국어로 1-2문장 설명)

예시:
태그: 고양이, 밈, 웃긴, 놀란 표정, 동물
설명: 놀란 표정을 짓고 있는 고양이 밈 이미지."""


def _dedupe_csv(s: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for t in (s or "").split(","):
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)


def _parse_response(response: str) -> tuple[str, str]:
    tags = ""
    description = ""
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("태그:"):
            tags = line[len("태그:"):].strip()
        elif line.startswith("설명:"):
            description = line[len("설명:"):].strip()
    if not tags and not description:
        description = response.strip()
    return _dedupe_csv(tags), description


def _image_to_data_url(image_path: Path) -> tuple[str, str, str]:
    """이미지를 base64로 인코딩. (mime, base64_str, data_url) 반환."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime or not mime.startswith("image/"):
        # 확장자 추론 실패 — 기본 jpeg로
        mime = "image/jpeg"
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return mime, b64, f"data:{mime};base64,{b64}"


class VLMProvider(Protocol):
    name: str

    def ensure_ready(self) -> None: ...
    def analyze(self, image_path: Path) -> tuple[str, str]: ...


# ───────────────────── Local Qwen ─────────────────────

# 로컬 모델 카탈로그 — 사용자가 선택 가능한 변종
# quant: None (FP16) | "bnb4" (bitsandbytes 4bit nf4, double quant)
LOCAL_MODELS = {
    "qwen2.5-vl-7b": {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "label": "Qwen2.5-VL-7B (FP16, ~16GB VRAM)",
        "vram_gb": 16,
        "quant": None,
    },
    "qwen2.5-vl-7b-bnb4": {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "label": "Qwen2.5-VL-7B 4bit (bnb, ~6GB VRAM)",
        "vram_gb": 6,
        "quant": "bnb4",
    },
    "qwen2.5-vl-3b": {
        "id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "label": "Qwen2.5-VL-3B (FP16, ~7GB VRAM)",
        "vram_gb": 7,
        "quant": None,
    },
}


class LocalQwenProvider:
    def __init__(self, model_key: str = "qwen2.5-vl-7b", device: str = "cuda"):
        if model_key not in LOCAL_MODELS:
            raise ValueError(f"알 수 없는 로컬 모델: {model_key}")
        self.model_key = model_key
        self.model_id = LOCAL_MODELS[model_key]["id"]
        self.device = device
        self.name = LOCAL_MODELS[model_key]["label"]
        self._model = None
        self._processor = None

    def ensure_ready(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        kwargs: dict = {"device_map": self.device}
        quant = LOCAL_MODELS[self.model_key]["quant"]
        if quant == "bnb4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def analyze(self, image_path: Path) -> tuple[str, str]:
        self.ensure_ready()
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=256)
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        response = self._processor.decode(generated, skip_special_tokens=True).strip()
        return _parse_response(response)


# ───────────────────── OpenAI ─────────────────────

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OpenAI API 키 필요")
        self.api_key = api_key
        self.model = model
        self.name = f"OpenAI {model}"
        self._client = None

    def ensure_ready(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)

    def analyze(self, image_path: Path) -> tuple[str, str]:
        self.ensure_ready()
        _, _, data_url = _image_to_data_url(image_path)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=512,
        )
        return _parse_response(resp.choices[0].message.content or "")


# ───────────────────── Anthropic ─────────────────────

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
        if not api_key:
            raise ValueError("Anthropic API 키 필요")
        self.api_key = api_key
        self.model = model
        self.name = f"Anthropic {model}"
        self._client = None

    def ensure_ready(self) -> None:
        if self._client is not None:
            return
        from anthropic import Anthropic
        self._client = Anthropic(api_key=self.api_key)

    def analyze(self, image_path: Path) -> tuple[str, str]:
        self.ensure_ready()
        mime, b64, _ = _image_to_data_url(image_path)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": mime, "data": b64,
                    }},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return _parse_response(text)


# ───────────────────── Gemini ─────────────────────

GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]


class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("Gemini API 키 필요")
        self.api_key = api_key
        self.model = model
        self.name = f"Gemini {model}"
        self._client = None

    def ensure_ready(self) -> None:
        if self._client is not None:
            return
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def analyze(self, image_path: Path) -> tuple[str, str]:
        self.ensure_ready()
        from google.genai import types
        mime, _, _ = _image_to_data_url(image_path)
        image_bytes = image_path.read_bytes()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                VLM_PROMPT,
            ],
        )
        return _parse_response(resp.text or "")


# ───────────────────── Factory ─────────────────────

def make_provider(settings: dict) -> VLMProvider:
    """settings dict에서 provider 인스턴스 생성.

    settings 형식:
        {
            "provider": "local" | "openai" | "anthropic" | "gemini",
            "local": {"model_key": "qwen2.5-vl-7b-awq", "device": "cuda"},
            "openai": {"api_key": "sk-...", "model": "gpt-4o-mini"},
            ...
        }
    """
    kind = settings.get("provider", "local")
    if kind == "local":
        cfg = settings.get("local", {})
        return LocalQwenProvider(
            model_key=cfg.get("model_key", "qwen2.5-vl-7b"),
            device=cfg.get("device", "cuda"),
        )
    if kind == "openai":
        cfg = settings.get("openai", {})
        return OpenAIProvider(api_key=cfg.get("api_key", ""), model=cfg.get("model", "gpt-4o-mini"))
    if kind == "anthropic":
        cfg = settings.get("anthropic", {})
        return AnthropicProvider(api_key=cfg.get("api_key", ""), model=cfg.get("model", "claude-haiku-4-5"))
    if kind == "gemini":
        cfg = settings.get("gemini", {})
        return GeminiProvider(api_key=cfg.get("api_key", ""), model=cfg.get("model", "gemini-2.0-flash"))
    raise ValueError(f"알 수 없는 provider: {kind}")


# ───────────────────── 설정 영속화 ─────────────────────

def load_settings(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 첫 실행 — VRAM 감지해서 적절한 디폴트 추천
    vram = detect_vram_gb()
    if vram is None:
        # GPU 없음 — API 모드 비워두고 사용자가 ⚙에서 키 입력 유도
        return {
            "provider": "openai",
            "openai": {"api_key": "", "model": "gpt-4o-mini"},
            "anthropic": {"api_key": "", "model": "claude-haiku-4-5"},
            "gemini": {"api_key": "", "model": "gemini-2.0-flash"},
        }
    if vram < 8:
        default_model = "qwen2.5-vl-7b-bnb4"  # 6GB
    elif vram < 12:
        default_model = "qwen2.5-vl-3b"      # 7GB
    else:
        default_model = "qwen2.5-vl-7b"      # 16GB
    return {"provider": "local", "local": {"model_key": default_model, "device": "cuda"}}


def save_settings(path: Path, settings: dict) -> None:
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def settings_for_display(settings: dict) -> dict:
    """API 키를 마스킹한 사본 (UI 전송용)."""
    out = json.loads(json.dumps(settings))  # deep copy
    for key in ("openai", "anthropic", "gemini"):
        if key in out and "api_key" in out[key]:
            k = out[key]["api_key"]
            out[key]["api_key_mask"] = (k[:6] + "•••" + k[-4:]) if len(k) > 12 else ("•••" if k else "")
            out[key]["api_key"] = ""  # 절대 평문 전송 X
    return out


TRANSLATE_CHAR_PROMPT = """다음은 Danbooru 스타일의 영어 캐릭터 태그 목록이다. 각각을 한국어 이름으로 번역해라.

규칙:
- 일본어/중국어 캐릭터명은 표준 한국어 표기로 (예: hakui_koyori → 하쿠이 코요리)
- 영문 이름은 한국어 음차 (예: hatsune_miku → 하츠네 미쿠)
- 언더스코어는 공백으로
- _(작품명) 같은 접미사는 무시하고 본명만 번역 (예: hakui_koyori_(1st_costume) → 하쿠이 코요리)
- 모르는 이름이면 해당 키에 null 사용
- 응답은 JSON 객체 하나로만: {"원본_영어_태그": "한국어 이름", ...}
- JSON 외에는 아무것도 출력하지 마

태그 목록:
"""


def translate_chars_to_ko(provider, names: list[str], batch_size: int = 40) -> dict[str, str]:
    """영어 캐릭터 태그 N개를 한국어로 번역해 dict로 반환.

    배치별로 텍스트 호출 (이미지 첨부 X). provider별로 텍스트-only 경로가 달라
    각자의 analyze() 대신 SDK를 직접 호출.
    """
    import json as _json
    result: dict[str, str] = {}
    if not names:
        return result
    provider.ensure_ready()

    for start in range(0, len(names), batch_size):
        batch = names[start:start + batch_size]
        prompt = TRANSLATE_CHAR_PROMPT + "\n".join(batch)
        try:
            text = _call_text(provider, prompt)
            parsed = _parse_translation_json(text, batch)
            result.update(parsed)
        except Exception:
            import traceback
            traceback.print_exc()
    return result


def _call_text(provider, prompt: str) -> str:
    """provider 종류에 따라 텍스트 한 번 호출. analyze는 이미지 첨부라 부적합."""
    name = provider.__class__.__name__
    if name == "OpenAIProvider":
        resp = provider._client.chat.completions.create(
            model=provider.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
    if name == "AnthropicProvider":
        resp = provider._client.messages.create(
            model=provider.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    if name == "GeminiProvider":
        resp = provider._client.models.generate_content(model=provider.model, contents=[prompt])
        return resp.text or ""
    if name == "LocalQwenProvider":
        # 로컬 Qwen은 텍스트만 호출도 멀티모달 메시지로 처리
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = provider._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = provider._processor(text=[text], return_tensors="pt").to(provider._model.device)
        with torch.no_grad():
            output_ids = provider._model.generate(**inputs, max_new_tokens=2048)
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return provider._processor.decode(generated, skip_special_tokens=True).strip()
    raise RuntimeError(f"알 수 없는 provider: {name}")


def _parse_translation_json(text: str, expected_keys: list[str]) -> dict[str, str]:
    """LLM이 반환한 JSON 텍스트에서 dict 추출. JSON 외 텍스트 섞여 와도 견디게."""
    import json as _json
    import re as _re
    # JSON 블록 추출
    m = _re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, _re.DOTALL)
    if not m:
        return {}
    try:
        data = _json.loads(m.group(0))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def with_retry(fn, max_attempts: int = 3, base_delay: float = 1.0):
    """rate limit / 일시적 네트워크 에러에 대해 exponential backoff 재시도.

    영구 에러(401/403, 잘못된 키 등)는 즉시 raise — 무한 재시도 방지.
    """
    import time
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # 영구 에러 — 재시도 무의미
            if any(k in msg for k in ("401", "403", "invalid api key", "permission denied", "unauthorized")):
                raise
            # 마지막 시도면 그냥 raise
            if attempt == max_attempts - 1:
                raise
            # backoff: 1s, 2s, 4s
            time.sleep(base_delay * (2 ** attempt))
    raise last_err  # unreachable


def detect_vram_gb() -> Optional[float]:
    """CUDA VRAM 총량 (GB). 감지 실패 시 None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory
        return round(total / (1024 ** 3), 1)
    except Exception:
        return None

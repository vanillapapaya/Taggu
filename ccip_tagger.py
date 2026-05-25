"""CCIP (Contrastive Anime Character Image Pre-training) 임베딩 추출.

deepghs/ccip_onnx의 ONNX 모델로 캐릭터 동일성 판단용 임베딩 추출.
WD14가 못 잡는 신규/희귀 캐릭터를 사용자 라벨로 학습하는 데 사용.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

CCIP_REPO = "deepghs/ccip_onnx"
CCIP_VARIANT = "ccip-caformer-24-randaug-pruned"  # 가성비 좋은 pruned 변종
CCIP_INPUT_SIZE = 384

# ImageNet 정규화 — CAFormer 백본은 표준 ImageNet preprocessing 사용
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_ccip(providers: Optional[list[str]] = None):
    """CCIP feature 모델 + cluster threshold 메타 로드. 첫 호출 시 ~150MB 다운로드."""
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    print(f"CCIP 로딩 중: {CCIP_REPO}/{CCIP_VARIANT}")
    feat_path = hf_hub_download(CCIP_REPO, f"{CCIP_VARIANT}/model_feat.onnx")
    cluster_path = hf_hub_download(CCIP_REPO, f"{CCIP_VARIANT}/cluster.json")

    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(feat_path, providers=providers)

    with open(cluster_path, encoding="utf-8") as f:
        cluster = json.load(f)

    # cluster.json은 OPTICS clustering용 threshold 등 — same/diff 판단 기준치
    # 보통 cluster["threshold"] = 동일 캐릭터 cosine 임계값
    print(f"CCIP 로딩 완료 (input={CCIP_INPUT_SIZE}x{CCIP_INPUT_SIZE})")
    return {"session": sess, "cluster": cluster}


def _preprocess(image_path: Path) -> np.ndarray:
    """CCIP 입력 전처리: 384x384, ImageNet 정규화, NCHW float32."""
    img = Image.open(image_path).convert("RGBA")
    # 투명 배경은 흰색으로
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")

    # 정사각 패딩 (CCIP는 aspect 변형보다 패딩이 robust)
    w, h = img.size
    s = max(w, h)
    padded = Image.new("RGB", (s, s), (255, 255, 255))
    padded.paste(img, ((s - w) // 2, (s - h) // 2))
    padded = padded.resize((CCIP_INPUT_SIZE, CCIP_INPUT_SIZE), Image.BICUBIC)

    arr = np.asarray(padded, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0)  # NCHW


def embed(image_path: Path, ccip: dict) -> np.ndarray:
    """이미지 한 장의 CCIP 임베딩 (L2-normalized float32 1D)."""
    sess = ccip["session"]
    arr = _preprocess(image_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    feat = sess.run([output_name], {input_name: arr})[0][0]  # (D,)
    # L2 정규화 — cosine 유사도 = dot product가 되도록
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """이미 L2-normalized라 dot product = cosine."""
    return float(np.dot(a, b))

# -*- mode: python ; coding: utf-8 -*-
"""Taggu PyInstaller spec — Mid 빌드 (CPU torch + CLIP + WD14 + API providers).

빌드: build.bat 또는 .venv-build\\Scripts\\pyinstaller taggu.spec --noconfirm
결과: dist/Taggu/Taggu.exe + 사이드 파일들 (~1GB)
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 데이터 파일 — 런타임에 EXE 옆에 위치해야 하는 것들
datas = [
    ('templates', 'templates'),
    ('character_aliases.json', '.'),
    ('icon.ico', '.'),
    ('icon.png', '.'),
]

# Jinja2 템플릿 / open_clip 가중치 메타 / fastapi 등 자동 수집
datas += collect_data_files('open_clip')
datas += collect_data_files('huggingface_hub')

# Lazy import / dynamic import 로 PyInstaller가 못 잡는 것들
hiddenimports = [
    # providers 모듈 — app에서 import하지만 클래스는 lazy
    'providers',
    # API SDK들 (providers.py에서 lazy import)
    'openai',
    'anthropic',
    'google.genai',
    'google.genai.types',
    # FastAPI / uvicorn 내부
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    # WD14 lazy 의존성
    'onnxruntime',
    # PIL 플러그인
    'PIL._tkinter_finder',
]

# 로컬 Qwen VLM 관련 — 이 빌드에는 포함 안 함 (API 모드 또는 별도 풀빌드)
excludes = [
    'transformers',
    'qwen_vl_utils',
    'bitsandbytes',
    'accelerate',
    'torchaudio',
    'tkinter',
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'sphinx',
]

a = Analysis(
    ['taggu_main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Taggu',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI mode — 콘솔 창 안 뜸
    disable_windowed_traceback=False,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Taggu',
)

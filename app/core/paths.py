"""프로그램 폴더 위치, Tesseract 실행 파일 찾기, 사용자 설정 저장."""
from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Optional

APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(APP_ROOT, "models")
DET_MODEL = os.path.join(MODELS_DIR, "ppocrv5_det.onnx")
TESSDATA_DIR = os.path.join(MODELS_DIR, "tessdata")
PROFILES_DIR = os.path.join(APP_ROOT, "profiles")
SETTINGS_PATH = os.path.join(APP_ROOT, "settings.json")

IS_WINDOWS = sys.platform.startswith("win")


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        cur = load_settings()
        cur.update(data)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def find_tesseract() -> Optional[str]:
    """설정 → PATH → 기본 설치 위치 순서로 tesseract 실행 파일을 찾는다."""
    cand = []
    s = load_settings().get("tesseract_cmd")
    if s:
        cand.append(s)
    w = shutil.which("tesseract")
    if w:
        cand.append(w)
    if IS_WINDOWS:
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
                     os.environ.get("LOCALAPPDATA", "")):
            if base:
                cand.append(os.path.join(base, "Tesseract-OCR", "tesseract.exe"))
    else:
        cand += ["/usr/bin/tesseract", "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]
    for c in cand:
        if c and os.path.isfile(c):
            return c
    return None


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def usable_tessdata_dir() -> str:
    """Windows용 Tesseract는 한글 등 비ASCII 경로의 언어 파일을 못 읽는 경우가 있다.
    프로그램 폴더 경로에 한글이 섞여 있으면 영문 경로(ProgramData)로 복사해서 쓴다."""
    if not IS_WINDOWS or _is_ascii(TESSDATA_DIR):
        return TESSDATA_DIR
    base = os.environ.get("ProgramData") or os.environ.get("PUBLIC") or r"C:\ProgramData"
    dst = os.path.join(base, "GameTextExtractor", "tessdata")
    try:
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(TESSDATA_DIR):
            if not f.endswith(".traineddata"):
                continue
            s, d = os.path.join(TESSDATA_DIR, f), os.path.join(dst, f)
            if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(s):
                shutil.copyfile(s, d)
        return dst
    except OSError:
        return TESSDATA_DIR


def available_languages() -> list:
    try:
        return sorted(f[:-12] for f in os.listdir(TESSDATA_DIR) if f.endswith(".traineddata"))
    except OSError:
        return []

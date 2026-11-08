"""설치 확인: python -m app.selftest"""
from __future__ import annotations

import os
import sys


def main() -> int:
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            msg = fn()
            print(f"  [정상] {name}" + (f" - {msg}" if msg else ""))
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [실패] {name} - {e}")

    print("설치 상태를 확인합니다.")
    check("Python", lambda: sys.version.split()[0])
    for mod in ("numpy", "cv2", "av", "onnxruntime", "openpyxl", "PIL", "kiwipiepy", "tkinter"):
        check(f"모듈 {mod}", lambda m=mod: getattr(__import__(m), "__version__", ""))

    from app.core import paths
    from app.core.ocr import LineOCR, TesseractRecognizer, TextDetector

    check("글자 위치 모델", lambda: (TextDetector(), os.path.basename(paths.DET_MODEL))[1])
    check("Tesseract", lambda: TesseractRecognizer("kor").version())
    check("한국어 언어 파일", lambda: ", ".join(paths.available_languages()))

    def ocr_test():
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        font = None
        for f in (r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\malgun.ttf",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
            if os.path.exists(f):
                font = ImageFont.truetype(f, 40)
                break
        if font is None:
            return "한글 글꼴이 없어 건너뜀"
        img = Image.new("RGB", (900, 140), (235, 235, 240))
        ImageDraw.Draw(img).text((30, 45), "설치가 정상적으로 끝났습니다.", font=font, fill=(40, 40, 50))
        arr = np.array(img)[:, :, ::-1].copy()
        lines = LineOCR("kor").read(arr)
        text = " ".join(l.text for l in lines)
        if "설치" not in text:
            raise RuntimeError(f"인식 결과가 이상합니다: {text!r}")
        return text

    check("글자 인식 시험", ocr_test)

    def kiwi_test():
        from app.core.postprocess import _kiwi_space
        r = _kiwi_space("컨베이어 라인", "옆에도")
        if r is None:
            raise RuntimeError("Kiwi를 불러오지 못했습니다")
        return "띄어쓰기 판단 가능"

    check("한국어 띄어쓰기 분석기", kiwi_test)
    print()
    print("모든 항목이 정상입니다." if ok else "실패한 항목이 있습니다. 사용설명서의 '문제 해결'을 확인하세요.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

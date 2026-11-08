"""명령줄 실행: python -m app.cli 영상.mp4 [영상2.mp4 ...] -p profiles/프로필.json [-o 출력폴더]"""
from __future__ import annotations

import argparse
import sys

from .core.pipeline import extract
from .core.profile import Profile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="게임 영상에서 텍스트를 뽑아 엑셀로 저장합니다.")
    ap.add_argument("videos", nargs="+", help="영상 파일 (여러 개 가능)")
    ap.add_argument("-p", "--profile", required=True, help="게임 프로필 JSON")
    ap.add_argument("-o", "--out", default=None, help="출력 폴더 (기본: 영상과 같은 폴더)")
    ap.add_argument("--debug", action="store_true", help="문제 분석용 상세 기록(_debug_readings.json) 저장")
    ap.add_argument("--engine", choices=("auto", "tesseract", "ppocr"), default="auto",
                    help="글자 인식 엔진 (기본 auto = 한국어면 PP-OCRv5 한국어 모델, 아니면 Tesseract)")
    ap.add_argument("--no-autocorrect", action="store_true", help="자주 틀리는 글자 자동 교정을 끈다 (비교 시험용)")
    a = ap.parse_args(argv)
    prof = Profile.load(a.profile)
    ocr = None
    for v in a.videos:
        print(f"▶ {v}")
        last = [-1]

        def prog(f, msg):
            p = int(f * 100)
            if p != last[0]:
                last[0] = p
                print(f"\r  [{p:3d}%] {msg}".ljust(90), end="", flush=True)
        res = extract(v, prof, a.out, progress=prog, ocr=ocr, debug=a.debug,
                      engine=a.engine, autocorrect=not a.no_autocorrect)
        review = sum(1 for r in res.records if r.needs_review)
        print(f"\n  완료: {len(res.records)}줄 (확인 필요 {review}줄) → {res.xlsx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

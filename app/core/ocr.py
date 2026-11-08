"""글자 위치 찾기(PP-OCRv5 검출 모델) + 글자 읽기(Tesseract).

1) 검출: PaddleOCR의 PP-OCRv5 텍스트 검출 모델(ONNX)로 글자가 있는 줄의 위치를 찾는다.
   배경 그림이 복잡해도 글자 줄만 정확히 잘라낼 수 있다.
2) 인식: 줄 단위로 잘라낸 이미지를 읽는다. 엔진은 둘 중 하나.
   - ppocr: PP-OCRv5 한국어 인식 모델(ONNX)로 컬러 줄 이미지를 그대로 읽는다(한국어 게임 기본).
   - tesseract: 흑백으로 정리한 뒤 Tesseract로 읽는다(그 밖의 언어, 또는 모델 파일이 없을 때).
3) 띄어쓰기: Tesseract는 한국어 띄어쓰기를 자주 틀리므로,
   실제 이미지에서 글자 사이의 빈 간격을 재서 띄어쓰기를 다시 만든다.
   (ppocr은 띄어쓰기도 모델이 직접 낸다.)
"""
from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import paths

# ---------------------------------------------------------------- 데이터 구조


@dataclass
class Line:
    text: str
    conf: float
    box: Tuple[int, int, int, int]  # x0, y0, x1, y1 (입력 이미지 기준 픽셀)

    @property
    def h(self) -> int:
        return self.box[3] - self.box[1]


class OcrError(RuntimeError):
    pass


# ---------------------------------------------------------------- 텍스트 검출


class TextDetector:
    """PP-OCRv5 DB 텍스트 검출 모델을 onnxruntime으로 직접 실행한다."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: str = paths.DET_MODEL, max_side: int = 1280,
                 bin_thresh: float = 0.3, box_thresh: float = 0.5, unclip_ratio: float = 1.6):
        import onnxruntime as ort

        if not os.path.isfile(model_path):
            raise OcrError(f"검출 모델 파일이 없습니다: {model_path}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.max_side = max_side
        self.bin_thresh = bin_thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self._lock = threading.Lock()

    def detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if img is None or img.size == 0:
            return []
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        nh = max(32, int(round(h * scale / 32)) * 32)
        nw = max(32, int(round(w * scale / 32)) * 32)
        x = cv2.resize(img, (nw, nh)).astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD
        x = x.transpose(2, 0, 1)[None]
        with self._lock:
            prob = self.sess.run(None, {self.inp: x})[0][0, 0]
        return self._boxes(prob, w / nw, h / nh, w, h)

    def _boxes(self, prob, sx, sy, w, h):
        bitmap = (prob > self.bin_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours[:1000]:
            x, y, bw, bh = cv2.boundingRect(c)
            if min(bw, bh) < 3:
                continue
            mask = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(mask, [c - [x, y]], 1)
            score = float(cv2.mean(prob[y:y + bh, x:x + bw], mask)[0])
            if score < self.box_thresh:
                continue
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            d = area * self.unclip_ratio / max(peri, 1e-6)
            x0, y0 = (x - d) * sx, (y - d) * sy
            x1, y1 = (x + bw + d) * sx, (y + bh + d) * sy
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
            if x1 - x0 >= 4 and y1 - y0 >= 6:
                out.append((x0, y0, x1, y1))
        return out


def group_lines(boxes: Sequence[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """단어 단위 상자들을 같은 높이끼리 묶어 '줄' 상자로 만든다.
    같은 높이라도 가로로 많이 떨어져 있으면 다른 줄로 본다."""
    rects = sorted(boxes, key=lambda r: (r[1] + r[3]) / 2)
    lines: List[List[int]] = []
    for r in rects:
        cy, rh = (r[1] + r[3]) / 2, r[3] - r[1]
        placed = False
        for L in lines:
            lcy, lh = (L[1] + L[3]) / 2, L[3] - L[1]
            same_row = (L[1] <= cy <= L[3]) or (r[1] <= lcy <= r[3])
            if not same_row:
                continue
            gap = max(r[0] - L[2], L[0] - r[2])
            if gap > 3.0 * max(rh, lh):
                continue
            L[0], L[1], L[2], L[3] = min(L[0], r[0]), min(L[1], r[1]), max(L[2], r[2]), max(L[3], r[3])
            placed = True
            break
        if not placed:
            lines.append(list(r))
    # 묶는 과정에서 겹치게 된 줄을 한 번 더 합친다
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                ov = min(a[3], b[3]) - max(a[1], b[1])
                if ov > 0.5 * min(a[3] - a[1], b[3] - b[1]) and max(a[0] - b[2], b[0] - a[2]) <= 3.0 * max(a[3] - a[1], b[3] - b[1]):
                    lines[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del lines[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(L) for L in sorted(lines, key=lambda L: (L[1], L[0]))]


# ---------------------------------------------------------------- 텍스트 인식


class TesseractRecognizer:
    """줄 이미지 한 장을 Tesseract로 읽는다 (이미지는 표준입력으로 넘겨 경로 문제를 피함)."""

    LINE_HEIGHT = 48     # 줄 이미지를 이 높이로 맞춘 뒤 읽는다 (실험상 가장 정확)
    SPACE_RATIO = 0.15   # 글자 사이 빈 간격이 줄 높이의 이 비율보다 넓으면 띄어쓰기

    def __init__(self, lang: str = "kor", tesseract_cmd: Optional[str] = None,
                 tessdata_dir: Optional[str] = None):
        self.cmd = tesseract_cmd or paths.find_tesseract()
        if not self.cmd:
            raise OcrError("Tesseract를 찾을 수 없습니다. 설치.bat을 다시 실행하거나, "
                           "프로그램 설정에서 tesseract.exe 위치를 지정해 주세요.")
        self.tessdata = tessdata_dir or paths.usable_tessdata_dir()
        self.lang = lang
        for code in lang.split("+"):
            if not os.path.isfile(os.path.join(self.tessdata, f"{code}.traineddata")):
                raise OcrError(f"언어 파일이 없습니다: {code}.traineddata (models/tessdata 폴더에 넣어 주세요)")
        # 한국어 문장 속 영문(플레이어 이름 등)이 깨져 보이면 '한국어+영어'로 한 번 더 읽는다
        codes = lang.split("+")
        self.alt_lang = None
        if "eng" not in codes and os.path.isfile(os.path.join(self.tessdata, "eng.traineddata")):
            self.alt_lang = lang + "+eng"
        self._flags = 0
        if paths.IS_WINDOWS:
            self._flags = 0x08000000  # CREATE_NO_WINDOW: 검은 창이 깜빡이지 않게

    def version(self) -> str:
        r = subprocess.run([self.cmd, "--version"], capture_output=True, creationflags=self._flags)
        out = (r.stdout or r.stderr).decode("utf-8", "replace").splitlines()
        return out[0] if out else "?"

    @classmethod
    def _scaled(cls, gray: np.ndarray) -> np.ndarray:
        scale = cls.LINE_HEIGHT / max(1, gray.shape[0])
        return cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)

    @staticmethod
    def _finish(b: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(b, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)

    @classmethod
    def prepare(cls, gray: np.ndarray, mode: str = "otsu") -> np.ndarray:
        """줄 이미지를 '흰 바탕 검은 글씨'로 정리한다.
        otsu: 전체 밝기 기준 (대부분의 대사창)
        bright / dark: 배경색이 섞여 있을 때 밝은 글씨 / 어두운 글씨만 골라냄"""
        g = cls._scaled(gray)
        if mode == "otsu":
            _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if (b > 0).mean() < 0.5:  # 어두운 바탕의 흰 글씨 → 반전
                b = 255 - b
            return cls._finish(b)
        if mode in ("white", "black"):
            # 순백/순흑 글씨만 남긴다 (밝은 그림 위의 흰 글씨 등)
            if mode == "white":
                T = max(200.0, float(np.percentile(g, 99)) - 30)
                b = np.where(g >= T, 0, 255).astype(np.uint8)
            else:
                T = min(60.0, float(np.percentile(g, 1)) + 30)
                b = np.where(g <= T, 0, 255).astype(np.uint8)
            return cls._finish(b)
        k = int(cls.LINE_HEIGHT * 0.4) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        op = cv2.MORPH_TOPHAT if mode == "bright" else cv2.MORPH_BLACKHAT
        h = cv2.morphologyEx(g, op, kernel)
        _, b = cv2.threshold(h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cls._finish(255 - b)

    def _run(self, img: np.ndarray, lang: Optional[str] = None) -> str:
        ok, png = cv2.imencode(".png", img)
        if not ok:
            return ""
        # 언어 파일 폴더를 따로 지정하면 기본 'tsv' 설정 파일을 못 찾으므로 -c로 직접 켠다
        args = [self.cmd, "stdin", "stdout", "--psm", "7", "-l", lang or self.lang,
                "--tessdata-dir", self.tessdata,
                "-c", "tessedit_create_txt=0", "-c", "tessedit_create_tsv=1"]
        try:
            r = subprocess.run(args, input=png.tobytes(), capture_output=True, timeout=60,
                               creationflags=self._flags)
        except subprocess.TimeoutExpired:
            return ""
        if r.returncode != 0:
            msg = r.stderr.decode("utf-8", "replace").strip()
            raise OcrError(f"Tesseract 실행 오류: {msg[:300]}")
        return r.stdout.decode("utf-8", "replace")

    RETRY_BELOW = 85.0  # 신뢰도가 이보다 낮으면 다른 전처리로 다시 읽어 본다

    def read_line(self, gray: np.ndarray) -> Tuple[str, float]:
        best = self._read_variant(gray, "otsu")
        # 신뢰도가 낮거나, 글자 뒤에 색 박스가 있어 흑백 변환이 뭉개졌으면(잉크 비율 과다)
        # 밝은 글씨/어두운 글씨 전용 전처리로 다시 읽고 '확실하게 읽힌 글자 수'가 많은 쪽을 고른다
        if best.conf < self.RETRY_BELOW or best.ink > 0.30:
            for mode in ("bright", "white", "dark", "black"):
                cand = self._read_variant(gray, mode)
                if _better(cand, best):
                    best = cand
                if best.conf >= 90:
                    break
        # 한국어 문장 속 영문(플레이어 이름 등)이 '[|' 같은 기호로 깨졌으면
        # 한국어+영어로 다시 읽어 '깨진 단어 자리만' 영문으로 바꾼다
        if self.alt_lang and any(self._LATIN_JUNK.search(w[0]) for w in best.raw):
            alt = self._read_variant(gray, best.mode, self.alt_lang)
            merged = self._merge_latin(best, alt)
            if merged is not None:
                best = merged
        return best.text, best.conf

    _LATIN_JUNK = re.compile(r"[\[\]|{}\\/]|^[1lI|!]{2,}$")
    _ASCII_WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'\-]*[.,!?~]*$")

    def _merge_latin(self, base: "_Read", alt: "_Read") -> Optional["_Read"]:
        words = list(base.words)
        changed = False
        for i, w in enumerate(words):
            if not self._LATIN_JUNK.search(w[0]):
                continue
            x0, x1 = w[2], w[2] + w[3]
            over = [a for a in alt.words if min(x1, a[2] + a[3]) - max(x0, a[2]) > 0.5 * min(w[3], a[3])]
            if over and all(self._ASCII_WORD.match(a[0]) for a in over):
                words[i] = (" ".join(a[0] for a in over), max(a[1] for a in over), x0, w[3])
                changed = True
        if not changed:
            return None
        return self._compose(words, base.img, base.ink, base.raw, base.mode)

    def _read_variant(self, gray: np.ndarray, mode: str, lang: Optional[str] = None) -> "_Read":
        img = self.prepare(gray, mode)
        ink_ratio = float((img[16:-16, 16:-16] < 128).mean()) if img.shape[0] > 32 else 0.0
        tsv = self._run(img, lang)
        words = []
        for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
            t = (row.get("text") or "").strip()
            try:
                conf = float(row.get("conf", -1))
            except ValueError:
                conf = -1
            if not t or conf < 0:
                continue
            words.append((t, conf, int(row["left"]), int(row["width"])))
        words.sort(key=lambda w: w[2])
        raw = list(words)
        # 줄 양 끝에 붙은 기호 한 글자짜리 잡음(아이콘 오인식)만 버린다 — 실제 글자는 지우지 않음
        while words and len(words[-1][0]) == 1 and not _MEANINGFUL.search(words[-1][0]) and words[-1][1] < 50:
            words.pop()
        while words and len(words[0][0]) == 1 and not _MEANINGFUL.search(words[0][0]) and words[0][1] < 50:
            words.pop(0)
        return self._compose(words, img, ink_ratio, raw, mode)

    def _compose(self, words, img, ink_ratio, raw, mode) -> "_Read":
        if not words:
            return _Read("", 0.0, ink_ratio, [], raw, img, mode)
        ink = (img < 128).any(axis=0)
        H = self.LINE_HEIGHT
        out = words[0][0]
        for a, b in zip(words, words[1:]):
            lo = int(a[2] + a[3] - 0.5 * H)
            hi = int(b[2] + 0.5 * H)
            lo, hi = max(lo, a[2]), min(hi, b[2] + b[3])
            gap = _longest_false_run(ink[lo:hi]) if hi > lo else 0
            out += (" " if gap > self.SPACE_RATIO * H else "") + b[0]
        n = sum(len(w[0]) for w in words)
        conf = sum(w[1] * len(w[0]) for w in words) / max(1, n)
        return _Read(out, conf, ink_ratio, words, raw, img, mode)


@dataclass
class _Read:
    text: str
    conf: float
    ink: float
    words: list
    raw: list
    img: np.ndarray
    mode: str


def _better(c: "_Read", b: "_Read") -> bool:
    """다른 전처리 결과(c)가 기존 결과(b)보다 확실히 나은지.
    신뢰도가 뚜렷이 높거나, 비슷한 신뢰도로 글자를 훨씬 많이 읽었을 때만 바꾼다 (잡음을 많이 읽는 경우 방지)."""
    nc, nb = len(_MEANINGFUL.findall(c.text)), len(_MEANINGFUL.findall(b.text))
    if nc == 0:
        return False
    if c.conf >= b.conf + 8 and nc >= 0.8 * nb:
        return True
    return c.conf >= 80 and nc >= 1.5 * max(1, nb) and b.conf - c.conf < 10


def _score(r: "_Read") -> float:
    return len(_MEANINGFUL.findall(r.text)) * r.conf


def _longest_false_run(arr: np.ndarray) -> int:
    best = run = 0
    for v in arr:
        run = 0 if v else run + 1
        if run > best:
            best = run
    return best


# ---------------------------------------------------------------- 정리 규칙

_DOTS = re.compile(r"(?:[.…·・‥]\s*){2,}")
_EDGE_JUNK = "|`'_\\¦"


def clean_text(s: str, ellipsis: str = "……") -> str:
    s = s.replace("​", "").strip()
    if ellipsis:
        s = _DOTS.sub(lambda m: ellipsis + (" " if m.group(0).endswith(" ") else ""), s)
        s = re.sub(r"[.,]{1,2}\s?(%s)" % re.escape(ellipsis), r"\1", s)  # ".,……" 같은 점 오인식 정리
    s = re.sub(r"\s+", " ", s)
    s = s.strip(_EDGE_JUNK + " ")
    s = re.sub(r"^[,.;:]+(?![.…])\s*", "", s)
    # 문장 끝 뒤에 붙은 1~2글자 영숫자 잡음 (예: "괜찮은 거야? 5.")
    s = re.sub(r"([.!?~…])\s+[0-9A-Za-z]{1,2}[.,]?$", r"\1", s)
    # 줄 맨 앞의 아이콘 오인식 (예: 위치 핀 → "@ ", "? ")
    s = re.sub(r"^[@#*•◆◇■□▶▷►※☆★♡♥?>=+©®]\s+", "", s)
    s = re.sub(r"\s+([,.!?~])", r"\1", s)
    return s.strip()


_MEANINGFUL = re.compile(r"[0-9A-Za-zㄱ-ㆎ가-힣぀-ヿ一-鿿]")


def is_meaningful(s: str) -> bool:
    if _MEANINGFUL.search(s):
        return True
    return bool(re.fullmatch(r"[.…·]{2,}[?!]*", s.replace(" ", "")))


# ---------------------------------------------------------------- 인식: PP-OCRv5 한국어 모델


class PaddleRecognizer:
    """PP-OCRv5 한국어 인식 모델(korean_PP-OCRv5_mobile_rec)을 onnxruntime으로 실행한다.

    줄 이미지(컬러) → 높이를 모델 입력에 맞춰 늘이고 → 글자 칸마다 확률 → CTC 풀이.
    글자 목록: 0번 = 빈칸(blank), 1..N = 사전 파일의 줄, N+1 = 띄어쓰기.
    신뢰도: 고른 글자들의 확률 평균 × 100 (Tesseract 신뢰도와 눈금이 다르다).

    모델만으로는 모자란 두 가지를 보탠다 (1편 정답표에서 잰 결과):
    - 말줄임표: 이 게임의 '……'(낮게 찍힌 작은 점)를 모델은 3~48% 확률로만 보고 빈칸에 진다.
      빈칸이 이긴 칸이라도 '…' 확률이 ELLIPSIS_MIN 이상이면 '…'로 본다.
    - 띄어쓰기: 모델은 띄어쓰기의 35%를 빠뜨린다(특히 쉼표 · 마침표 뒤).
      Tesseract 쪽과 같은 방법으로, 이웃한 두 글자 사이의 빈 세로줄 폭을 재서 넓으면 띄운다."""

    wants_color = True
    HEIGHT = 48          # 입력 높이가 모델에 고정돼 있지 않을 때 쓰는 값 (PP-OCRv5 학습 높이)
    MAX_WIDTH = 3200     # 아주 긴 줄의 상한 (메모리 보호)
    ELLIPSIS_MIN = 0.03  # 0이면 끈다
    SPACE_RATIO = TesseractRecognizer.SPACE_RATIO

    def __init__(self, model_path: str = paths.REC_MODEL, dict_path: str = paths.REC_DICT):
        import onnxruntime as ort

        for f, what in ((model_path, "인식 모델"), (dict_path, "인식 모델 글자 사전")):
            if not os.path.isfile(f):
                raise OcrError(f"{what} 파일이 없습니다: {f}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        h = inp.shape[2] if len(inp.shape) == 4 else None
        self.height = h if isinstance(h, int) and h > 0 else self.HEIGHT
        with open(dict_path, encoding="utf-8") as f:
            chars = [ln.rstrip("\r\n") for ln in f]
        while chars and chars[-1] == "":
            chars.pop()
        self.chars = [""] + chars + [" "]
        self.model_path = model_path
        n_out = self.sess.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(self.chars):
            raise OcrError(f"인식 모델 출력({n_out}개)과 글자 사전({len(self.chars)}개, 빈칸 · 띄어쓰기 포함)이 맞지 않습니다.")

    def version(self) -> str:
        return os.path.splitext(os.path.basename(self.model_path))[0]

    def _resized(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        nw = int(min(self.MAX_WIDTH, max(self.height, np.ceil(self.height * w / max(1, h)))))
        return cv2.resize(img, (nw, self.height), interpolation=cv2.INTER_CUBIC if self.height > h else cv2.INTER_AREA)

    def prepare(self, img: np.ndarray) -> np.ndarray:
        x = self._resized(img).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return x.transpose(2, 0, 1)[None]

    def _ctc(self, probs: np.ndarray):
        """probs: (T, 글자 수) → [(글자, 칸 번호, 확률)]. 같은 글자 연속은 하나로, 빈칸은 버린다."""
        idx = probs.argmax(axis=1)
        p = probs.max(axis=1)
        e = self.chars.index("…") if (self.ELLIPSIS_MIN > 0 and "…" in self.chars) else -1
        if e > 0:
            weak = (idx == 0) & (probs[:, e] >= self.ELLIPSIS_MIN)
            idx = np.where(weak, e, idx)
            p = np.where(weak, probs[:, e], p)
        out, prev = [], -1
        for t, k in enumerate(idx):
            if k != prev and k != 0 and k < len(self.chars):
                out.append((self.chars[k], t, float(p[t])))
            prev = k
        return out

    def decode(self, probs: np.ndarray, ink: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """ink: 모델 입력 폭과 같은 길이의 '세로줄에 글자 획이 있나' 배열. 주면 넓은 빈틈에 띄어쓰기를 넣는다."""
        cs = self._ctc(probs)
        if not cs:
            return "", 0.0
        T = probs.shape[0]
        out = cs[0][0]
        for (a, ta, _), (b, tb, _) in zip(cs, cs[1:]):
            if ink is not None and a != " " and b != " ":
                step = len(ink) / T
                lo, hi = int((ta + 0.5) * step), int((tb + 0.5) * step)
                if hi > lo and _longest_false_run(ink[lo:hi]) > self.SPACE_RATIO * self.height:
                    out += " "
            out += b
        conf = 100.0 * float(np.mean([c[2] for c in cs if c[0] != " "] or [0.0]))
        return re.sub(r" {2,}", " ", out).strip(), conf

    def ink_columns(self, img: np.ndarray) -> np.ndarray:
        g = self._resized(img)
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (b > 0).mean() < 0.5:  # 어두운 바탕의 흰 글씨 → 반전 (Tesseract 쪽 otsu와 같다)
            b = 255 - b
        return (b < 128).any(axis=0)

    def read_line(self, img: np.ndarray) -> Tuple[str, float]:
        y = self.sess.run(None, {self.input_name: self.prepare(img)})[0]
        return self.decode(y[0], self.ink_columns(img))


# ---------------------------------------------------------------- 묶음


class LineOCR:
    """영역 이미지 한 장 → 줄 목록."""

    ENGINES = ("auto", "tesseract", "ppocr")

    @staticmethod
    def pick_engine(lang: str) -> str:
        """auto: 한국어(+영어) 게임이고 한국어 인식 모델이 있으면 ppocr, 아니면 tesseract.
        1편 정답표: 글자 오류율 1.94% → 1.25%, 글자가 틀린 줄(문장부호 제외) 31 → 11, 처리 시간 144초 → 44초."""
        codes = set(lang.split("+"))
        if "kor" in codes and codes <= {"kor", "eng"} and os.path.isfile(paths.REC_MODEL) and os.path.isfile(paths.REC_DICT):
            return "ppocr"
        return "tesseract"

    def __init__(self, lang: str = "kor", tesseract_cmd: Optional[str] = None, engine: str = "auto"):
        from concurrent.futures import ThreadPoolExecutor

        self.detector = TextDetector()
        if engine == "auto":
            engine = self.pick_engine(lang)
        if engine == "ppocr":
            self.recognizer = PaddleRecognizer()
        elif engine == "tesseract":
            self.recognizer = TesseractRecognizer(lang=lang, tesseract_cmd=tesseract_cmd)
        else:
            raise OcrError(f"알 수 없는 인식 엔진: {engine} ({', '.join(self.ENGINES)} 중 하나)")
        self.engine = engine
        self._pool = ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2))))

    def detect_rows(self, img_bgr: np.ndarray,
                    ignore: Sequence[Tuple[int, int, int, int]] = ()) -> List[Tuple[int, int, int, int]]:
        """글자 줄 위치만 빠르게 찾는다 (글자 읽기 없이)."""
        boxes = self.detector.detect(img_bgr)
        if ignore:
            boxes = [b for b in boxes if not _mostly_inside(b, ignore)]
        return group_lines(boxes)

    def read(self, img_bgr: np.ndarray, min_conf: float = 0.0, ellipsis: str = "……",
             ignore: Sequence[Tuple[int, int, int, int]] = (), rows=None) -> List[Line]:
        if rows is None:
            rows = self.detect_rows(img_bgr, ignore)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        if not rows:
            return []
        med_h = float(np.median([r[3] - r[1] for r in rows]))
        jobs = []
        for (x0, y0, x1, y1) in rows:
            # 다른 줄보다 훨씬 작은 상자(장식 무늬, 아이콘)는 글자가 아닐 가능성이 높다
            if len(rows) > 1 and (y1 - y0) < 0.6 * med_h and (x1 - x0) < 2.5 * (y1 - y0):
                continue
            pad = max(4, int(0.15 * (y1 - y0)))
            src = img_bgr if getattr(self.recognizer, "wants_color", False) else gray
            crop = src[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
            if crop.size:
                jobs.append(((x0, y0, x1, y1), crop))
        # 줄마다 Tesseract를 따로 실행하므로 여러 줄을 동시에 읽어 시간을 줄인다
        results = list(self._pool.map(lambda j: self.recognizer.read_line(j[1]), jobs))
        out: List[Line] = []
        for (box, _), (text, conf) in zip(jobs, results):
            text = clean_text(text, ellipsis)
            if not text or conf < min_conf or not is_meaningful(text):
                continue
            out.append(Line(text, conf, box))
        return out


def find_dot_line(gray: np.ndarray, ref: Line, ellipsis: str = "……") -> Optional[Line]:
    """'......'만 있는 줄은 글자 검출 모델이 놓치므로, 기준 줄(보통 화자 이름) 아래에서
    같은 크기의 점이 일정 간격으로 3개 이상 늘어선 곳을 찾아 말줄임표 줄로 만든다."""
    h = max(8, ref.h)
    H, W = gray.shape
    y0, y1 = ref.box[3], min(H, ref.box[3] + int(3.5 * h))
    x0, x1 = max(0, ref.box[0] - 8 * h), min(W, ref.box[2] + 8 * h)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    _, b = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for fg in (b, 255 - b):
        n, _, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
        dots = []
        for i in range(1, n):
            x, y, w, hh, area = st[i]
            if 0.05 * h <= w <= 0.3 * h and 0.05 * h <= hh <= 0.3 * h and 0.5 <= w / max(1, hh) <= 2 \
                    and area >= 0.45 * w * hh:
                dots.append((cen[i][0], cen[i][1], x, y, w, hh))
        dots.sort()
        best: list = []
        for d in dots:
            run = [d]
            for e in dots:
                if e[0] <= run[-1][0]:
                    continue
                if abs(e[1] - d[1]) <= 0.12 * h and 0.1 * h <= e[0] - run[-1][0] <= 0.6 * h:
                    run.append(e)
            if len(run) > len(best):
                best = run
        if len(best) >= 3:
            bx0 = min(d[2] for d in best) + x0
            bx1 = max(d[2] + d[4] for d in best) + x0
            by1 = max(d[3] + d[5] for d in best) + y0
            return Line(ellipsis or "......", 75.0, (int(bx0), int(by1 - 0.8 * h), int(bx1), int(by1 + 0.1 * h)))
    return None


def _mostly_inside(b, rects, frac=0.6) -> bool:
    area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    for r in rects:
        iw = min(b[2], r[2]) - max(b[0], r[0])
        ih = min(b[3], r[3]) - max(b[1], r[1])
        if iw > 0 and ih > 0 and iw * ih >= frac * area:
            return True
    return False

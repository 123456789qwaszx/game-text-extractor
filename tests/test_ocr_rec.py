"""PP-OCRv5 인식기의 풀이(CTC)와 입력 준비 — 모델 파일 없이 확인한다."""
import unittest

import numpy as np

from app.core.ocr import PaddleRecognizer


def recognizer(chars):
    r = object.__new__(PaddleRecognizer)  # 모델을 열지 않고 풀이만 시험
    r.chars = [""] + list(chars) + [" "]
    r.height = 48
    return r


def probs(seq, n):
    """글자 번호 줄 → 그 번호에 확률 0.9를 준 (T, n) 표."""
    p = np.full((len(seq), n), 0.1 / (n - 1), np.float32)
    for t, k in enumerate(seq):
        p[t, k] = 0.9
    return p


class DecodeTests(unittest.TestCase):
    def test_repeats_collapse_and_blank_separates(self):
        r = recognizer("가나다")  # 1=가 2=나 3=다 4=띄어쓰기
        text, conf = r.decode(probs([1, 1, 0, 1, 2, 2, 4, 3, 0], 5))
        self.assertEqual(text, "가가나 다")
        self.assertAlmostEqual(conf, 90.0, places=3)

    def test_spaces_at_edges_are_trimmed_and_runs_squeezed(self):
        r = recognizer("가나")  # 3=띄어쓰기
        text, _ = r.decode(probs([3, 1, 3, 0, 3, 2, 3], 4))
        self.assertEqual(text, "가 나")

    def test_all_blank_is_empty_with_zero_confidence(self):
        r = recognizer("가")
        self.assertEqual(r.decode(probs([0, 0, 0], 3)), ("", 0.0))


class PrepareTests(unittest.TestCase):
    def test_height_fixed_aspect_kept_values_in_minus1_to_1(self):
        r = recognizer("가")
        r.MAX_WIDTH = 3200
        img = np.zeros((24, 120, 3), np.uint8)
        img[:, 60:] = 255
        x = r.prepare(img)
        self.assertEqual(x.shape, (1, 3, 48, 240))
        self.assertAlmostEqual(float(x.min()), -1.0)
        self.assertAlmostEqual(float(x.max()), 1.0)

    def test_gray_input_becomes_three_channels(self):
        r = recognizer("가")
        x = r.prepare(np.full((48, 48), 128, np.uint8))
        self.assertEqual(x.shape, (1, 3, 48, 48))


class EllipsisTests(unittest.TestCase):
    def test_weak_ellipsis_beats_blank(self):
        # 1편: '……'에 모델이 준 확률이 3~48%라 빈칸에 져서 사라졌다
        r = recognizer("가나…")  # 3=…
        p = probs([1, 0, 0, 2], 5)
        p[1] = [0.6, 0.0, 0.0, 0.4, 0.0]
        self.assertEqual(r.decode(p)[0], "가…나")

    def test_off_when_floor_is_zero(self):
        r = recognizer("가나…")
        r.ELLIPSIS_MIN = 0
        p = probs([1, 0, 2], 5)
        p[1] = [0.6, 0.0, 0.0, 0.4, 0.0]
        self.assertEqual(r.decode(p)[0], "가나")


class SpacingTests(unittest.TestCase):
    def test_wide_ink_gap_between_letters_is_a_space(self):
        # 모델이 빠뜨린 띄어쓰기(쉼표 뒤 등)를 글자 사이 빈 세로줄 폭으로 되살린다
        r = recognizer("가나다")
        p = probs([1, 0, 2, 0, 0, 0, 3, 0], 5)   # 8칸 × 8px = 폭 64
        ink = np.ones(64, bool)
        ink[30:45] = False                       # 나(칸2) 와 다(칸6) 사이 15px 빈틈 > 0.15 × 48
        ink[12:16] = False                       # 가 와 나 사이 4px — 붙여 쓴다
        self.assertEqual(r.decode(p, ink)[0], "가나 다")

    def test_model_spaces_are_kept(self):
        r = recognizer("가나")  # 3=띄어쓰기
        p = probs([1, 3, 2], 4)
        self.assertEqual(r.decode(p, np.ones(24, bool))[0], "가 나")


if __name__ == "__main__":
    unittest.main()

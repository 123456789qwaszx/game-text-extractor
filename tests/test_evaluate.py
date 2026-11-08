"""점수 도구 — 지어낸 짧은 정답표와 결과로 짝짓기 · 점수를 시험한다(엑셀 · OCR 없음)."""
import unittest

from app.evaluate import Golden, Result, align, evaluate, levenshtein, norm


def G(no, t, text, kind="대사", spk="가"):
    return Golden(no, t, kind, spk, text)


def R(no, t, text, kind="대사", spk="가", review=""):
    return Result(no, t, kind, spk, text, review)


class EvaluateTests(unittest.TestCase):
    def test_norm_unifies_ellipsis_quotes_and_spaces(self):
        self.assertEqual(norm("“아, 음......” 그래…… 좋아 ..."), '"아,음…"그래…좋아…')

    def test_levenshtein(self):
        self.assertEqual(levenshtein("이력서군", "이럭서군"), 1)
        self.assertEqual(levenshtein("", "abc"), 3)

    def test_missing_extra_duplicate_and_scores(self):
        gold = [G(1, 10, "안녕하세요 반갑습니다"), G(2, 20, "오늘은 날씨가 좋네요"), G(3, 30, "……", spk="나"),
                G(4, 40, "첫 면접", kind="장제목", spk="")]
        res = [R(1, 10.5, "안녕하세요 반갑습니다"),
               R(2, 20.2, "오늘은 날씨가 좋내요", review="인식 신뢰도 낮음"),
               R(3, 21.0, "오늘은 날씨가"),                    # 타자기 도중에 한 번 더 읽음 — 중복
               R(4, 33.0, "월 아그리", kind="나레이션", spk="")]  # 잡음
        s = evaluate(gold, res)
        self.assertEqual((s["짝"], s["재현율"], s["정밀도"]), (2, 0.5, 0.5))
        self.assertEqual(s["헛줄"], {"중복": 1, "잡음": 1})
        self.assertEqual(s["종류별 재현율"], {"대사": "2/3", "장제목": "0/1"})
        self.assertEqual(s["편집 거리"], 1)
        self.assertEqual(s["확인 표시"]["표시된 줄 중 실제 오류"], "1/1")

    def test_of_two_readings_the_closer_one_is_the_pair(self):
        s = evaluate([G(1, 10, "오늘은 날씨가 좋네요")], [R(1, 10, "오늘은 날씨가 좋내요"), R(2, 11, "오늘은 날씨가 좋네요")])
        self.assertEqual((s["편집 거리"], s["헛줄"]["중복"]), (0, 1))

    def test_far_apart_lines_never_pair(self):
        pairs = align([G(1, 10, "같은 글")], [R(1, 30, "같은 글")])
        self.assertEqual(sorted(p for p in pairs if None not in p), [])

    def test_speaker_accuracy_counts_dialogue_only(self):
        gold = [G(1, 1, "가나다라마바", spk="후유카"), G(2, 5, "속으로 하는 말", kind="주인공속말", spk="")]
        res = [R(1, 1, "가나다라마바", spk="추유카"), R(2, 5, "속으로 하는 말", kind="나레이션", spk="")]
        s = evaluate(gold, res)
        self.assertEqual((s["화자 정확도"], s["화자 판정 줄"]), (0.0, 1))
        self.assertEqual(s["형태 (정답 → 결과)"], {"대사 → 대사": 1, "주인공속말 → 나레이션": 1})


if __name__ == "__main__":
    unittest.main()

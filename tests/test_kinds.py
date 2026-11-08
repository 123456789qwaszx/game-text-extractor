"""형태 규칙 · 화자 이름 정리 — 1편 · 3편B 정답표에서 나온 사례."""
import unittest

from app.core.ocr import Line, is_meaningful
from app.core.postprocess import Reading, clean_speaker, rule_kind

RULES = [
    {"kind": "주인공속말", "from": "나레이션", "speaker": False, "text_left_min": 0.28, "mark": "창 밝기", "mark_max": 120},
    {"kind": "주인공대사", "from": "나레이션", "speaker": False, "text_left_min": 0.28},
]


def rd(left_px, brightness):
    body = [Line("본문", 99.0, (left_px, 690, left_px + 400, 725))]
    return Reading(0, 1.0, 0, (0, 0, 1, 1), body, True, marks={"창 밝기": brightness}, frame_w=1920), body


class RuleKindTests(unittest.TestCase):
    def test_portrait_in_a_dark_box_is_inner_voice(self):
        r, body = rd(678, 29)  # 본문 시작 0.35 · 창 밝기 29
        self.assertEqual(rule_kind(RULES, r, "나레이션", False, body), "주인공속말")

    def test_portrait_in_a_light_box_is_protagonist_speech(self):
        r, body = rd(678, 224)
        self.assertEqual(rule_kind(RULES, r, "나레이션", False, body), "주인공대사")

    def test_text_at_the_usual_place_keeps_the_area_kind(self):
        r, body = rd(406, 224)  # 0.21 — 초상 없는 지문
        self.assertEqual(rule_kind(RULES, r, "나레이션", False, body), "")

    def test_rules_only_touch_their_own_base_kind(self):
        r, body = rd(678, 29)  # 장 제목 영역의 글도 가운데라 0.28을 넘는다
        self.assertEqual(rule_kind(RULES, r, "장제목", False, body), "")


class UnknownSpeakerTests(unittest.TestCase):
    def test_question_mark_nameplate_is_a_speaker(self):
        # 3편B: 이름표 '???' 대사 7줄이 화자 없는 나레이션이 됐다
        self.assertTrue(is_meaningful("???"))
        self.assertEqual(clean_speaker("???"), "???")
        self.assertEqual(clean_speaker("? ??"), "???")

    def test_other_speakers_are_trimmed_as_before(self):
        self.assertEqual(clean_speaker("· 닉스 ]"), "닉스")


if __name__ == "__main__":
    unittest.main()

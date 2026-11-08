"""화면 종류: 메신저 · 카드 선택 · UI — 3편B · 1편 정답표에서 나온 사례를 줄인 것."""
import unittest

from app.core.ocr import Line
from app.core.postprocess import Reading, build_records
from app.core.profile import AREA, SPEAKER, Profile, Region

W, H = 1920, 886


def profile():
    p = Profile(regions=[
        Region(SPEAKER, (0.21273, 0.68423, 0.41039, 0.76046)),
        Region(AREA, (0.06, 0.68, 0.83, 0.97), label="대사", label_no_speaker="나레이션"),
        Region(AREA, (0.31, 0.10, 0.75, 0.92), label="메신저", mode="messenger", header=(0.36, 0.12, 0.60, 0.235)),
        Region(AREA, (0.10, 0.20, 0.90, 0.95), label="선택지", mode="cards", header=(0.70, 0.86, 0.92, 0.95)),
    ])
    p.choice_marks = ["기존 선택"]
    p.choice_question = r"\?$"
    p.ui_phrases = ["스토리 입장"]
    return p


def L(text, x0, y0, x1, y1, tone=-1.0, conf=98.0):
    return Line(text, conf, (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)), tone)


def R(t, area, lines):
    return Reading(0, t, area, (0, 0, W, H), lines, True, frame_w=W)


def run(readings):
    p = profile()
    return build_records(readings, p, [(z, z.to_px(W, H)) for z in p.zones], frame_size=(W, H))


class MessengerTests(unittest.TestCase):
    def chat(self, t, bubbles):
        return R(t, 1, [L("닉스", 0.37, 0.137, 0.41, 0.19)] + bubbles)

    def test_each_bubble_once_with_side_as_speaker(self):
        a = L("정말 고마워, 닉스… 선배?", 0.40, 0.30, 0.68, 0.33, tone=140)
        b = L("굳이 선배라고 부르지 않아도 돼", 0.39, 0.40, 0.68, 0.435, tone=15)
        recs = run([self.chat(204.2, [a]), self.chat(206.1, [a, b]), self.chat(207.5, [a, b]),
                    R(206.1, 0, [L("굳이 선배라고 부르지", 0.21, 0.70, 0.5, 0.74)])])  # 대사창에 걸린 말풍선 조각
        self.assertEqual([(r.kind, r.speaker, r.text) for r in recs],
                         [("메신저", "나", "정말 고마워, 닉스… 선배?"), ("메신저", "닉스", "굳이 선배라고 부르지 않아도 돼")])

    def test_garbled_read_of_a_sliding_bubble_is_dropped(self):
        ok = L("아까 단합식 때는 깜짝 놀랐어", 0.40, 0.50, 0.60, 0.535, tone=120)
        bad = L("-이깨 단합식 때는 감 직 늘 렸 어", 0.40, 0.50, 0.60, 0.535, tone=94, conf=70)
        recs = run([self.chat(210.2, [bad]), self.chat(210.5, [ok]), self.chat(211.1, [ok])])
        self.assertEqual([r.text for r in recs], ["아까 단합식 때는 깜짝 놀랐어"])

    def test_picture_popup_with_digits_in_header_is_not_a_messenger(self):
        # 3편B 193.8초: 휴대폰 그림 팝업의 '000'을 상대 이름으로 읽어 그 순간의 대사가 버려졌다
        recs = run([R(193.8, 1, [L("000", 0.37, 0.15, 0.42, 0.22, conf=55)]),
                    R(193.9, 0, [L("저기, 혹시……", 0.21, 0.78, 0.31, 0.82)])])
        self.assertEqual([r.kind for r in recs], ["나레이션"])


class CardsTests(unittest.TestCase):
    def test_one_line_per_card_screen(self):
        q = L("어떤 걸 선택해야 하지?", 0.73, 0.886, 0.87, 0.923)
        cards = [L("직감", 0.20, 0.60, 0.24, 0.64), L("돈이나 휴가로 보상해 준다면 괜찮다", 0.14, 0.70, 0.30, 0.73),
                 L("분석", 0.41, 0.60, 0.45, 0.64), L("절대 안 된다!", 0.36, 0.70, 0.50, 0.73), L("기존 선택", 0.40, 0.80, 0.46, 0.83)]
        recs = run([R(290.4, 2, [q] + cards), R(291.0, 2, [q] + cards), R(293.0, 2, [q])])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].kind, "선택지")
        self.assertEqual(recs[0].text, "직감: 돈이나 휴가로 보상해 준다면 괜찮다\n분석: 절대 안 된다!")
        self.assertTrue(any("기존 선택" in n for n in recs[0].notes))

    def test_no_question_mark_no_card_screen(self):
        # 1편 0.1초: 스테이지 목록의 글이 카드처럼 가로로 늘어섰다
        recs = run([R(0.1, 2, [L("01: 특별 면접", 0.15, 0.4, 0.3, 0.44), L("02: 나이팅게일 빵", 0.5, 0.4, 0.65, 0.44),
                               L("로그: 거래", 0.75, 0.88, 0.85, 0.92)])])
        self.assertEqual([r.kind for r in recs if r.kind == "선택지"], [])


class UiTests(unittest.TestCase):
    def test_ui_window_at_start_hides_everything_before_it(self):
        recs = run([R(0.2, 0, [L("밤에 드리운 불빛", 0.06, 0.88, 0.2, 0.93)]),
                    R(0.6, 0, [L("스토리 입장", 0.56, 0.71, 0.64, 0.75), L("미지근한 물은 끓어올라", 0.3, 0.45, 0.7, 0.5)]),
                    R(5.0, 0, [L("면접관", 0.22, 0.70, 0.3, 0.74), L("어서 와.", 0.21, 0.78, 0.3, 0.82)])])
        self.assertEqual([(r.kind, r.text) for r in recs], [("UI", "스토리 입장"), ("대사", "어서 와.")])


class NumbersOnlyTests(unittest.TestCase):
    def test_battle_hud_numbers_are_not_dialogue(self):
        recs = run([R(125.0, 0, [L("30 9", 0.4, 0.80, 0.45, 0.84)]), R(130.0, 0, [L("면접관", 0.22, 0.70, 0.3, 0.74), L("……", 0.21, 0.78, 0.25, 0.82)])])
        self.assertEqual([(r.speaker, r.text) for r in recs], [("면접관", "……")])


if __name__ == "__main__":
    unittest.main()

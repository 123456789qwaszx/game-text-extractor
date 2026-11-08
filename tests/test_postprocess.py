"""줄 합치기 규칙 — 실제 결함에서 나온 사례."""
import unittest

from app.core.postprocess import Record, relation


def rec(t, text, speaker="", t_last=None):
    return Record(t=t, t_last=t if t_last is None else t_last, stream=(0, "label:장소/시간"), kind="장소/시간",
                  speaker=speaker, text=text, conf=90.0, shot="", area_box=(0, 0, 1, 1), stable=True)


class RelationTests(unittest.TestCase):
    def test_a_similar_card_much_later_is_a_new_card(self):
        # 1편: 10:10 카드(11.1초까지) 뒤 55.7초의 10:30 카드가 앞 카드에 합쳐져 사라졌다
        p = rec(10.2, "10:10 운월\n30일\n필리에 온실 구역 아그리 유니온 면접실", t_last=11.1)
        c = rec(55.7, "10:30 운월\n30일\n필리에 온실 구역 아그리 유니온 면접실")
        self.assertEqual(relation(p, c), "different")

    def test_the_same_text_read_again_soon_is_the_same_line(self):
        p = rec(10.2, "10:10 운월\n30일\n필리에 온실 구역", t_last=10.8)
        c = rec(11.1, "10:10운월\n30일\n필리에 은실 구역")
        self.assertEqual(relation(p, c), "same")

    def test_an_ellipsis_only_line_is_not_the_start_of_a_later_line(self):
        # 1편: 43.1초 면접관 '……' 뒤 16초 만의 '이력서를 보니…'가 '자라는 중'으로 합쳐져 둘 다 사라졌다
        p = rec(43.2, "……", speaker="면접관", t_last=45.9)
        c = rec(59.8, "이력서를 보니 공용 노바어 고급 자격증", speaker="면접관")
        self.assertEqual(relation(p, c), "different")

    def test_text_drawn_right_after_an_ellipsis_still_grows(self):
        p = rec(10.0, "……", t_last=10.2)
        c = rec(10.5, "……그래서")
        self.assertEqual(relation(p, c), "grow")

    def test_an_ellipsis_line_does_not_grow_into_unrelated_text(self):
        # 3편B 30.7초: 코하쿠 '……' 뒤 잠깐 비친 잡음 'IKOC'가 '자라는 중'으로 합쳐져 줄이 사라졌다
        p = rec(30.7, "……", speaker="코하쿠", t_last=33.2)
        c = rec(33.7, "IKOC")
        self.assertEqual(relation(p, c), "different")


if __name__ == "__main__":
    unittest.main()

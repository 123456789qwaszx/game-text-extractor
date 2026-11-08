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


if __name__ == "__main__":
    unittest.main()

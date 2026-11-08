"""정답표 ↔ 추출 결과 엑셀 — 줄마다 짝을 짓고 항목별 점수를 낸다.

  python -m app.evaluate <정답표 golden.json> <결과 엑셀> [--report 비교표.csv] [--json 점수.json]

정답표(gte-golden/1): lines[] = {no, t, t_first, kind, speaker, text, note}
  kind: 대사 · 주인공대사 · 주인공속말 · 선택지 · 장소카드 · 장제목 · UI
  t: 글자가 다 나온 첫 순간(초)

짝짓기: 정답표와 결과를 각자 시간순으로 두고, 글(화자 칸 + 원문)의 닮음으로 순서를 지키며 1:1로 짝짓는다
(동적 계획법). 시각이 MAX_GAP초 넘게 떨어지면 짝이 될 수 없다. 짝이 없는 결과 줄은 헛줄이다 —
이웃한 정답 줄과 닮았으면 '중복', 아니면 '잡음'.

점수
  재현율      정답 줄 중 짝이 있는 비율(종류별)
  정밀도      결과 줄 중 짝이 있는 비율
  화자        짝이 있는 '대사'에서 화자 칸이 정답과 같은 비율
  글자 오류율  짝이 있는 대사 · 주인공 줄의 편집 거리 합 / 정답 글자 수 합(공백은 빼고 센다)
  시각 오차    결과 시간 − 정답 t
  확인 표시    노란 칸('확인'에 '자동 교정' 말고 다른 까닭)이 실제 오류(글자 오류 > 0)를 얼마나 가려내나
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

MAX_GAP = 8.0        # 초 — 이보다 멀면 짝이 아니다
MIN_SIM = 0.4        # 이보다 덜 닮으면 짝이 아니다
TEXT_KINDS = ("대사", "주인공대사", "주인공속말")

_ELLIPSIS = re.compile(r"(?:\.{2,}|…+|·{2,})")
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})


def norm(s: str) -> str:
    """비교용 — 말줄임표 · 따옴표 모양을 하나로, 공백을 뺀다."""
    s = _ELLIPSIS.sub("…", (s or "").translate(_QUOTES))
    return re.sub(r"\s+", "", s)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class Golden:
    no: int
    t: float
    kind: str
    speaker: str
    text: str


@dataclass
class Result:
    no: int
    t: float
    kind: str
    speaker: str
    text: str
    review: str = ""
    conf: Optional[float] = None

    @property
    def flagged(self) -> bool:
        reasons = [r.strip() for r in self.review.split(",") if r.strip()]
        return any(not r.startswith("자동 교정") for r in reasons)


def parse_time(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    parts = [float(p) for p in str(v).strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def load_golden(path: str) -> List[Golden]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return [Golden(l["no"], float(l["t"]), l["kind"], l.get("speaker", ""), l["text"]) for l in doc["lines"]]


def load_results(path: str) -> List[Result]:
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True, data_only=True)["텍스트"]
    rows = ws.iter_rows(values_only=True)
    head = [str(h or "").strip() for h in next(rows)]
    col = {h: i for i, h in enumerate(head)}
    out = []
    for r in rows:
        if r[col["No"]] is None:
            continue
        g = lambda k: r[col[k]] if k in col else None
        out.append(Result(int(g("No")), parse_time(g("시간")), str(g("형태") or ""), str(g("화자") or ""),
                          str(g("원문") or ""), str(g("확인") or ""), g("신뢰도")))
    return out


def pair_sim(g: Golden, r: Result) -> float:
    if abs(r.t - g.t) > MAX_GAP:
        return 0.0
    # 선택지 · 메신저는 글이 화자 칸으로 새기도 하므로 화자 칸 + 원문과도 견준다
    return max(sim(g.text, r.text), sim(g.text, r.speaker + r.text))


def align(gold: Sequence[Golden], res: Sequence[Result]) -> List[Tuple[Optional[int], Optional[int]]]:
    """순서를 지키는 1:1 짝짓기. → [(정답 인덱스 | None, 결과 인덱스 | None)] 시간순."""
    n, m = len(gold), len(res)
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]       # 0 짝 · 1 정답 건너뜀 · 2 결과 건너뜀
    for i in range(1, n + 1):
        back[i][0] = 1
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, how = S[i - 1][j], 1
            if S[i][j - 1] > best:
                best, how = S[i][j - 1], 2
            s = pair_sim(gold[i - 1], res[j - 1])
            if s >= MIN_SIM and S[i - 1][j - 1] + s > best:
                best, how = S[i - 1][j - 1] + s, 0
            S[i][j], back[i][j] = best, how
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        how = back[i][j]
        if how == 0:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif how == 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


def evaluate(gold: Sequence[Golden], res: Sequence[Result]) -> dict:
    gold = sorted(gold, key=lambda g: g.t)
    res = sorted(res, key=lambda r: r.t)
    pairs = align(gold, res)
    matched = {gi: ri for gi, ri in pairs if gi is not None and ri is not None}
    by_res = {ri: gi for gi, ri in matched.items()}
    rows = []

    recall: Dict[str, List[int]] = {}
    for gi, g in enumerate(gold):
        recall.setdefault(g.kind, [0, 0])
        recall[g.kind][1] += 1
        if gi in matched:
            recall[g.kind][0] += 1

    # 헛줄 — 이웃 정답 줄(시각 ±MAX_GAP)과 닮았으면 중복
    extras = {"중복": 0, "잡음": 0}
    extra_rows = []
    for ri, r in enumerate(res):
        if ri in by_res:
            continue
        near = [g for g in gold if abs(g.t - r.t) <= MAX_GAP]
        best = max(((max(sim(g.text, r.text), sim(g.text, r.speaker + r.text)), g) for g in near),
                   key=lambda x: x[0]) if near else (0, None)
        kind = "중복" if best[0] >= 0.5 else "잡음"
        extras[kind] += 1
        extra_rows.append((r, kind, best[1]))

    spk_ok = spk_n = 0
    edits = chars = 0
    dts = []
    flag = {"flag_err": 0, "flag_ok": 0, "noflag_err": 0, "noflag_ok": 0}
    kinds: Dict[Tuple[str, str], int] = {}
    for gi, ri in sorted(matched.items()):
        g, r = gold[gi], res[ri]
        kinds[(g.kind, r.kind)] = kinds.get((g.kind, r.kind), 0) + 1
        dts.append(r.t - g.t)
        if g.kind == "대사":
            spk_n += 1
            spk_ok += norm(r.speaker) == norm(g.speaker)
        e = None
        if g.kind in TEXT_KINDS:
            a, b = norm(g.text), norm(r.text)
            e = levenshtein(a, b)
            edits += e
            chars += len(a)
            key = ("flag" if r.flagged else "noflag") + ("_err" if e > 0 else "_ok")
            flag[key] += 1
        rows.append({"정답": g.no, "결과": r.no, "정답 시각": g.t, "결과 시각": r.t, "정답 형태": g.kind, "결과 형태": r.kind,
                     "정답 화자": g.speaker, "결과 화자": r.speaker, "정답 원문": g.text, "결과 원문": r.text,
                     "편집 거리": e, "확인": r.review})
    for gi, g in enumerate(gold):
        if gi not in matched:
            rows.append({"정답": g.no, "결과": None, "정답 시각": g.t, "정답 형태": g.kind, "정답 화자": g.speaker,
                         "정답 원문": g.text, "판정": "빠짐"})
    for r, kind, g in extra_rows:
        rows.append({"정답": None, "결과": r.no, "결과 시각": r.t, "결과 형태": r.kind, "결과 화자": r.speaker,
                     "결과 원문": r.text, "판정": "헛줄 · " + kind, "닮은 정답": g.no if g else None})
    rows.sort(key=lambda x: x.get("정답 시각", x.get("결과 시각", 0)) or 0)

    adt = [abs(d) for d in dts]
    fe = flag["flag_err"] + flag["noflag_err"]
    ff = flag["flag_err"] + flag["flag_ok"]
    return {
        "정답 줄": len(gold), "결과 줄": len(res), "짝": len(matched),
        "재현율": round(len(matched) / max(1, len(gold)), 3),
        "정밀도": round(len(matched) / max(1, len(res)), 3),
        "종류별 재현율": {k: f"{a}/{b}" for k, (a, b) in sorted(recall.items())},
        "헛줄": extras,
        "화자 정확도": round(spk_ok / max(1, spk_n), 3), "화자 판정 줄": spk_n,
        "글자 오류율": round(edits / max(1, chars), 4), "글자 수": chars, "편집 거리": edits,
        "틀린 글자가 있는 줄": f"{fe}/{fe + flag['flag_ok'] + flag['noflag_ok']}",
        "시각 오차": {"중앙값": round(statistics.median(dts), 2) if dts else None,
                    "절댓값 중앙값": round(statistics.median(adt), 2) if adt else None,
                    "±1초 안": f"{sum(a <= 1.0 for a in adt)}/{len(adt)}"},
        "확인 표시": {"표시된 줄 중 실제 오류": f"{flag['flag_err']}/{ff}", "오류 줄 중 표시됨": f"{flag['flag_err']}/{fe}"},
        "형태 (정답 → 결과)": {f"{a} → {b or '(빈칸)'}": n for (a, b), n in sorted(kinds.items())},
        "_rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("golden")
    ap.add_argument("result")
    ap.add_argument("--report", help="줄마다 비교표(CSV, 엑셀에서 열림)")
    ap.add_argument("--json", help="점수(JSON)")
    a = ap.parse_args(argv)
    score = evaluate(load_golden(a.golden), load_results(a.result))
    rows = score.pop("_rows")
    for k, v in score.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"{k}: {v}")
    if a.report:
        cols = ["판정", "정답", "결과", "정답 시각", "결과 시각", "정답 형태", "결과 형태", "정답 화자", "결과 화자",
                "정답 원문", "결과 원문", "편집 거리", "확인", "닮은 정답"]
        with open(a.report, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"비교표 → {a.report}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(score, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

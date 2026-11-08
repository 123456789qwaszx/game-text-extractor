"""OCR 결과(읽은 순간들) → 최종 대사 목록.

- 줄을 영역(화자/분류/무시)에 배정
- 자동 줄바꿈으로 나뉜 줄을 한 문장으로 연결
- 타자기 효과로 여러 번 읽힌 같은 대사를 하나로 합침
- 화자 이름 오인식 보정, 확인이 필요한 줄 표시
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ocr import Line
from .profile import IGNORE, LABEL, SPEAKER, Profile, Region

Box = Tuple[int, int, int, int]


@dataclass
class Reading:
    rid: int
    t: float
    area_idx: int
    area_box: Box               # 프레임 기준 영역 좌표
    lines: List[Line]           # 프레임 기준 좌표로 변환된 줄
    stable: bool                # 화면이 멈춘 순간에 읽었는지(True) / 강제로 읽었는지(False)
    shot: str = ""              # 저장된 스크린샷 경로


@dataclass
class Record:
    t: float
    t_last: float
    stream: Tuple[int, str]
    kind: str
    speaker: str
    text: str
    conf: float
    shot: str
    area_box: Box
    stable: bool
    support: int = 1
    closed: bool = False        # 본문이 지워진 것을 확인함 → 이후 결과는 새 대사
    notes: List[str] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    variants: list = field(default_factory=list)  # 같은 대사로 합쳐진 읽기 결과들

    def add_variant(self, c: "Record") -> None:
        self.variants.append((c.text, c.conf, c.shot, c.lines, c.speaker, c.kind))

    def finalize(self) -> None:
        """같은 대사로 합쳐진 읽기 결과들로 최종 원문을 정한다.
        완성된 길이의 결과들 중 서로 가장 비슷한 것(대표)을 고르고, 3개 이상이면 글자별 다수결로 오인식을 고친다."""
        if not self.variants:
            return
        top = max(len(key(v[0])) for v in self.variants)
        pool = [v for v in self.variants if len(key(v[0])) >= 0.9 * top]
        if len(pool) >= 3:
            sims = [sum(ratio(a[0], b[0]) for b in pool) + a[1] / 1000.0 for a in pool]
            best = pool[int(np.argmax(sims))]
            text = vote(best[0], [v[0] for v in pool if v is not best])
        else:
            best = max(pool, key=lambda v: (v[1] / 100.0) * len(key(v[0])))
            text = best[0]
        self.text, self.conf, self.shot, self.lines = text, best[1], best[2], best[3]
        spk = Counter(v[4] for v in self.variants if v[4])
        if spk:
            self.speaker = spk.most_common(1)[0][0]
            self.kind = next(v[5] for v in self.variants if v[4] == self.speaker)

    @property
    def review(self) -> str:
        return ", ".join(dict.fromkeys(self.notes))

    @property
    def needs_review(self) -> bool:
        """사람이 확인해야 하는 줄인지 (자동 교정 안내만 있는 줄은 제외)."""
        return any(not n.startswith("자동 교정") for n in self.notes)


# ---------------------------------------------------------------- 영역 배정

def _anchor(line: Line) -> Tuple[float, float]:
    x0, y0, x1, y1 = line.box
    return x0 + 0.3 * (y1 - y0), (y0 + y1) / 2


def assign(lines: Sequence[Line], zones: Sequence[Tuple[Region, Box]]):
    speaker: List[Line] = []
    labels: Dict[str, List[Line]] = {}
    body: List[Line] = []
    for ln in lines:
        ax, ay = _anchor(ln)
        hit: Optional[Region] = None
        for z, (x0, y0, x1, y1) in zones:
            if x0 <= ax <= x1 and y0 <= ay <= y1:
                hit = z
                break
        if hit is None:
            body.append(ln)
        elif hit.type == IGNORE:
            continue
        elif hit.type == SPEAKER:
            speaker.append(ln)
        elif hit.type == LABEL:
            labels.setdefault(hit.label or "분류", []).append(ln)
    return speaker, labels, body


# ---------------------------------------------------------------- 줄 연결

_END_PUNCT = re.compile(r"[.!?~,…」』)\]\"'”’]$")
# 다음 줄이 이런 조사·어미로 시작하면 앞 줄 끝 단어가 잘린 것(글자 단위 줄바꿈)으로 보고 붙여 쓴다
_JOSA = ("은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "와", "과", "로", "고", "서",
         "며", "면", "요", "죠", "지", "게", "니", "야", "나", "랑", "께", "엔", "든", "써",
         "에서", "에게", "한테", "으로", "까지", "부터", "처럼", "보다", "라고", "이라", "이랑",
         "이나", "에는", "에도", "에선", "으로서", "으로써", "이다", "였다", "었다", "니까", "는데")
_JOSA_START = re.compile(r"^(%s)(?=$|[\s.,!?~…'\"’”)])" % "|".join(sorted(_JOSA, key=len, reverse=True)))


class _Spacer:
    """한국어 형태소 분석기(Kiwi)로 줄바꿈 지점의 띄어쓰기를 판단한다. 없으면 조사 규칙으로 대신한다."""
    _kiwi = None
    _tried = False

    @classmethod
    def get(cls):
        if not cls._tried:
            cls._tried = True
            try:
                from kiwipiepy import Kiwi
                cls._kiwi = Kiwi()
            except Exception:  # noqa: BLE001
                cls._kiwi = None
        return cls._kiwi


def _kiwi_space(left: str, right: str) -> Optional[bool]:
    kiwi = _Spacer.get()
    if kiwi is None:
        return None
    lw = left.split(" ")[-2:]
    rw = right.split(" ")[:2]
    L, R = " ".join(lw), " ".join(rw)
    n = len(L.replace(" ", ""))
    try:
        out = kiwi.space(L + R, reset_whitespace=False)
    except Exception:  # noqa: BLE001
        return None
    cnt = 0
    for i, ch in enumerate(out):
        if ch != " ":
            cnt += 1
            if cnt == n:
                return i + 1 < len(out) and out[i + 1] == " "
    return None


# Tesseract가 이 게임 글꼴에서 자주 헷갈리는 글자 → 바른 후보 (한국어 언어 모델 점수로 판단)
_CONFUSION = {"올": "을", "블": "불", "혼": "흔", "뚱": "뚫", "웅": "응", "꿀": "꿇", "꿈": "꿇", "끊": "끓",
              "를": "들", "물": "를", "랜": "렌", "쏟": "쓸", "멋": "떳", "켓": "겠", "럼": "렴", "율": "을",
              "흘": "을", "틀": "를", "빚": "빛", "즐": "줄", "름": "를", "엠": "앵", "매": "꽤", "럽": "렵"}


def lm_correct(text: str, margin: float = 4.0):
    """흔한 오인식 글자를 바꿔 보고, 한국어 형태소 분석 점수가 확실히 좋아질 때만 고친다.
    (예: 마법올→마법을, 블량→불량, 뚱고→뚫고) 고친 목록을 함께 돌려준다."""
    kiwi = _Spacer.get()
    if kiwi is None or not text:
        return text, []

    def score(s: str) -> float:
        try:
            return kiwi.analyze(s, top_n=1)[0][1]
        except Exception:  # noqa: BLE001
            return float("-inf")

    words = text.split(" ")
    fixes = []
    for i in range(len(words)):
        w = words[i]
        for j, ch in enumerate(w):
            alt = _CONFUSION.get(ch)
            if not alt:
                continue
            lo, hi = max(0, i - 2), min(len(words), i + 3)
            ctx = words[lo:hi]
            cand = w[:j] + alt + w[j + 1:]
            a = " ".join(ctx)
            b = " ".join(ctx[:i - lo] + [cand] + ctx[i - lo + 1:])
            if score(b) - score(a) > margin:
                fixes.append(f"{w}→{cand}")
                words[i] = w = cand
    return " ".join(words), fixes


def join_lines(lines: Sequence[Line], full_at, mode: str, korean: bool = True) -> str:
    """자동 줄바꿈으로 나뉜 줄을 한 문장으로 잇는다.
    auto: 앞 줄이 끝까지 꽉 찼으면(글자 단위 줄바꿈) 한국어 분석기로 띄어쓰기를 판단하고,
          꽉 차지 않았으면(의도한 줄바꿈) 띄어 쓴다."""
    lines = sorted(lines, key=lambda l: (l.box[1], l.box[0]))
    if not lines:
        return ""
    out = lines[0].text
    for prev, cur in zip(lines, lines[1:]):
        if mode == "space":
            sep = " "
        elif mode == "none":
            sep = ""
        elif _END_PUNCT.search(prev.text) or not full_at(prev):
            sep = " "
        else:
            d = _kiwi_space(prev.text, cur.text) if korean else None
            if d is None:
                d = not (korean and _JOSA_START.match(cur.text))
            sep = " " if d else ""
        out += sep + cur.text
    return out


# ---------------------------------------------------------------- 비교

_KEY_DROP = re.compile(r"[\s.,!?~…·'\"“”‘’\-–—:;()\[\]「」『』<>]")


def key(s: str) -> str:
    return _KEY_DROP.sub("", s)


def ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def vote(base: str, others: Sequence[str]) -> str:
    """대표 결과(base)의 글자마다, 다른 결과들과 맞춰 본 뒤 가장 많이 나온 글자를 고른다."""
    votes = [Counter({ch: 1}) for ch in base]
    for t in others:
        sm = SequenceMatcher(None, base, t, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
                for k in range(i2 - i1):
                    votes[i1 + k][t[j1 + k]] += 1
    return "".join(v.most_common(1)[0][0] for v in votes)


def relation(p: Record, c: Record) -> str:
    """앞 레코드(p)와 새로 읽은 결과(c)의 관계.
    same: 같은 대사 / grow: 타자기 효과로 더 길어짐 / shrink: 사라지는 중 / different: 새 대사"""
    if p.speaker and c.speaker and ratio(p.speaker, c.speaker) < 0.5:
        return "different"
    a, b = key(p.text), key(c.text)
    gap = c.t - p.t_last
    # 한참 뒤에 나온 글은 별개의 대사로 본다 — '같은 글' 판정보다 먼저 본다.
    # (장소/시간 카드는 시각 숫자만 다른 채로 몇십 초 뒤에 다시 나온다: 10:10 카드 뒤의 10:30 카드가 앞 카드에 합쳐졌다)
    if gap > 15.0:
        return "different"
    if a == b or (min(len(a), len(b)) >= 4 and ratio(a, b) >= 0.88):
        return "same"
    if len(a) < len(b):
        if not a:
            return "grow"
        # 읽는 순간 마지막 글자가 반쯤 그려져 있을 수 있으므로 앞부분 위주로 비교
        if len(a) >= 4 and gap <= 6.0 and ratio(a, b[:len(a)]) >= 0.7:
            return "grow"
        if len(a) < 4 and gap <= 2.0 and b.startswith(a):
            return "grow"
    if len(b) < len(a) and gap <= 6.0:
        if not b:
            return "shrink"
        if len(b) >= 4 and ratio(b, a[:len(b)]) >= 0.7:
            return "shrink"
        if len(b) < 4 and a.startswith(b):
            return "shrink"
    return "different"


_SPK_EDGE = re.compile(r"^[^0-9A-Za-z\u3131-\u318E\uAC00-\uD7A3\u3040-\u30FF\u4E00-\u9FFF]+|[^0-9A-Za-z\u3131-\u318E\uAC00-\uD7A3\u3040-\u30FF\u4E00-\u9FFF?!)]+$")


def _hangul_ratio(s: str) -> float:
    m = re.findall(r"[0-9A-Za-z\uAC00-\uD7A3\u3131-\u318E]", s)
    if not m:
        return 0.0
    return sum(1 for ch in m if "\uAC00" <= ch <= "\uD7A3" or "\u3131" <= ch <= "\u318E") / len(m)


def clean_speaker(s: str) -> str:
    return _SPK_EDGE.sub("", s.strip()).strip()


# ---------------------------------------------------------------- 메인

def build_records(readings: Sequence[Reading], profile: Profile,
                  zones_px: Sequence[Tuple[Region, Box]]) -> List[Record]:
    areas = profile.areas
    korean = "kor" in profile.language
    # 1) 줄 배정 + 줄 오른쪽 끝 통계 (같은 위치에서 시작하는 줄끼리 묶어 '꽉 찬 줄'의 기준을 잡는다)
    staged = []
    rights: Dict[Tuple[int, int], List[int]] = {}
    for rd in readings:
        speaker, labels, body = assign(rd.lines, zones_px)
        speaker = [l for l in speaker if (len(key(l.text)) >= 2 or l.conf >= 90)
                   and (not korean or re.search(r"[\uAC00-\uD7A3]|[A-Za-z]{2,}", l.text))]
        staged.append((rd, speaker, labels, body))
        for ln in body:
            rights.setdefault((rd.area_idx, ln.box[0] // 24), []).append(ln.box[2])
    max_right: Dict[Tuple[int, int], float] = {}
    for (ai, b), v in rights.items():
        pool = v + rights.get((ai, b - 1), []) + rights.get((ai, b + 1), [])
        max_right[(ai, b)] = float(np.percentile(pool, 95))

    def full_checker(area_idx):
        def full_at(ln: Line) -> bool:
            mr = max_right.get((area_idx, ln.box[0] // 24))
            return mr is not None and ln.box[2] >= mr - 1.5 * max(1, ln.h)
        return full_at

    # 2) 읽은 순간 → 후보 레코드
    cands: List[Record] = []
    for rd, speaker, labels, body in staged:
        area = areas[rd.area_idx]
        if body or speaker:
            sk = (rd.area_idx, "body")
            spk = clean_speaker(" ".join(l.text for l in sorted(speaker, key=lambda l: (l.box[1], l.box[0]))))
            text = join_lines(body, full_checker(rd.area_idx), profile.line_join, korean)
            used = list(speaker) + list(body)
            conf = min(l.conf for l in used)
            kind = (area.label or "대사") if spk else (area.label_no_speaker or "나레이션")
            cands.append(Record(rd.t, rd.t, sk, kind, spk, text, conf, rd.shot, rd.area_box,
                                rd.stable, lines=used))
        for name, lns in labels.items():
            lns = sorted(lns, key=lambda l: (l.box[1], l.box[0]))
            text = "\n".join(l.text for l in lns)
            cands.append(Record(rd.t, rd.t, (rd.area_idx, "label:" + name), name, "", text,
                                min(l.conf for l in lns), rd.shot, rd.area_box, rd.stable, lines=lns))

    # 3) 흐름(stream)별로 시간순으로 합치기
    merged: List[Record] = []
    last: Dict[Tuple[int, str], Record] = {}
    for c in sorted(cands, key=lambda r: r.t):
        p = last.get(c.stream)
        rel = relation(p, c) if (p is not None and not p.closed) else "different"
        if rel in ("same", "grow"):
            p.support += 1
            p.t_last = c.t
            p.stable = p.stable or c.stable
            if rel == "grow":  # 타자기 효과: 더 긴 쪽이 최신 상태
                p.variants = [v for v in p.variants if len(key(v[0])) >= len(key(c.text)) * 0.9]
            p.add_variant(c)
            p.finalize()
            continue
        if rel == "shrink":
            p.t_last = c.t
            if not key(c.text):
                p.closed = True
            continue
        c.add_variant(c)
        merged.append(c)
        last[c.stream] = c

    # 3-1) 장소/시간 카드처럼 날아 들어오는 연출: 짧은 시간 안의 같은 분류 결과는 가장 완전한 것 하나로
    merged = _merge_label_bursts(merged)

    # 4) 잡음 제거: 화면 전환 중에 한 번만 스친 낮은 신뢰도의 결과
    out: List[Record] = []
    for r in merged:
        if not key(r.text) and not r.speaker:
            continue
        if not r.speaker and len(key(r.text)) <= 2 and r.conf < profile.review_confidence:
            continue
        if korean and r.conf < profile.review_confidence and _hangul_ratio(r.text) < 0.5:
            continue  # 한국어 게임인데 한글이 거의 없는 낮은 신뢰도 결과 = 배경 무늬 오인식
        if not r.stable and r.support < 2:
            if r.conf < profile.review_confidence:
                continue
            # 화면 전환 중 한 번 읽힌 결과가 앞뒤의 제대로 읽힌 대사와 비슷하면 중복으로 보고 버린다
            near = [o for o in merged if o is not r and o.stream == r.stream and abs(o.t - r.t) <= 5.0
                    and (o.stable or o.support >= 2)]
            if any(ratio(key(o.text), key(r.text)) >= 0.6 for o in near):
                continue
            r.notes.append("화면 전환 중에 읽음")
        out.append(r)

    # 5) 화자 이름 확인 표시: 드물게 나온 이름이 자주 나온 이름과 한 글자만 다르면
    #    오인식일 수도, 실제로 다른 인물일 수도 있으므로 바꾸지 않고 '확인'만 요청한다
    counts = Counter(r.speaker for r in out if r.speaker)
    suspect: Dict[str, str] = {}
    for s_, n in counts.items():
        for s2, n2 in counts.items():
            if s2 != s_ and n2 >= max(3, 3 * n) and len(s2) == len(s_) and ratio(s_, s2) >= 0.6:
                suspect[s_] = s2
                break
    for r in out:
        if r.speaker in suspect:
            r.notes.append(f"화자명 확인 ('{suspect[r.speaker]}' 오인식일 수 있음)")

    # 5-1) 한국어 언어 모델로 흔한 오인식 글자 교정
    if korean:
        for r in out:
            fixed, fixes = lm_correct(r.text)
            if fixes:
                r.text = fixed
                r.notes.append("자동 교정: " + ", ".join(fixes[:4]))

    # 6) 사용자 치환 규칙 (예: 플레이어 이름 → {PLAYER}, 자주 틀리는 고유명사 교정)
    reps = [(a, b) for a, b in (profile.replacements or []) if a]
    if reps:
        for r in out:
            for a, b in reps:
                r.text = r.text.replace(a, b)
                r.speaker = r.speaker.replace(a, b)

    # 7) 확인 필요 표시
    for r in out:
        if not key(r.text):
            r.notes.insert(0, "본문 인식 안 됨")
        elif r.conf < profile.review_confidence:
            r.notes.insert(0, "인식 신뢰도 낮음")
        if r.stream[1] == "body" and key(r.text) and not _SENT_END.search(r.text.strip()):
            r.notes.append("문장 끝 확인 (중간에 넘어갔을 수 있음)")
    return out


_SENT_END = re.compile(r"[.!?~…」』)\]\"'”’♪♡♥*—\-]$|[다요죠네까야지군나어아]$")


def _merge_label_bursts(records: List[Record], window: float = 4.0) -> List[Record]:
    out: List[Record] = []
    groups: Dict[Tuple[int, str], Record] = {}
    for r in records:
        if not r.stream[1].startswith("label:"):
            out.append(r)
            continue
        g = groups.get(r.stream)
        if g is not None and r.t - g.t_last <= window:
            g.t_last = r.t
            g.support += r.support
            g.stable = g.stable or r.stable
            if (r.conf / 100.0) * len(key(r.text)) > (g.conf / 100.0) * len(key(g.text)):
                g.text, g.conf, g.shot, g.lines = r.text, r.conf, r.shot, r.lines
            continue
        groups[r.stream] = r
        out.append(r)
    # 카드가 날아오는 도중 일부 글자만 일반 텍스트로 읽힌 것 제거 (예: '4일')
    labels = [r for r in out if r.stream[1].startswith("label:")]
    keep = []
    for r in out:
        if r.stream[1] == "body" and not r.speaker and len(key(r.text)) <= 6:
            k = key(r.text)
            near = [L for L in labels if abs(r.t - L.t) <= window]
            if any(k in key(L.text) or max((ratio(k, key(part)) for part in L.text.split("\n")), default=0) >= 0.5
                   for L in near):
                continue
        keep.append(r)
    return keep

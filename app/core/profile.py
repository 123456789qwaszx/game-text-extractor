"""게임별 설정(프로필).

좌표는 모두 화면 크기에 대한 비율(0~1)로 저장한다.
녹화 해상도가 달라도(720p / 1080p / 4K) 같은 프로필을 그대로 쓸 수 있다.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

PROFILE_VERSION = 1

# 영역 종류
AREA = "area"          # 글자를 읽을 영역 (변화 감지 + OCR 대상)
SPEAKER = "speaker"    # 화자 이름이 나오는 자리
LABEL = "label"        # 별도 분류로 뽑을 자리 (예: 장소/시간 카드)
IGNORE = "ignore"      # 무시할 자리 (깜빡이는 아이콘, 고정 UI 글자 등)

ZONE_TYPES = (SPEAKER, LABEL, IGNORE)
TYPE_NAMES = {
    AREA: "텍스트 영역",
    SPEAKER: "화자 이름",
    LABEL: "별도 분류",
    IGNORE: "무시",
}

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1 (비율)


@dataclass
class Region:
    type: str
    rect: Rect
    label: str = ""  # AREA: 화자가 있을 때 형태 이름 / LABEL: 분류 이름
    label_no_speaker: str = ""  # AREA 전용: 화자가 없을 때 형태 이름
    mode: str = ""   # AREA 전용: "" = 대사창 · "messenger" = 메신저 화면(말풍선마다 한 줄) · "cards" = 카드 선택 화면
    header: Optional[Tuple[float, float, float, float]] = None  # 메신저: 상대 이름이 나오는 머리 자리(비율)

    def to_px(self, w: int, h: int) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.rect
        return (max(0, int(round(x0 * w))), max(0, int(round(y0 * h))),
                min(w, int(round(x1 * w))), min(h, int(round(y1 * h))))


@dataclass
class Profile:
    name: str = "새 프로필"
    version: int = PROFILE_VERSION
    language: str = "kor"            # Tesseract 언어 코드 (kor, eng, jpn, kor+eng ...)
    sample_fps: float = 10.0          # 초당 검사할 프레임 수
    stable_sec: float = 0.2           # 이 시간 동안 화면이 안 바뀌면 '대사 완성'으로 판단
    change_threshold: float = 0.05    # 변화 민감도: 영역 픽셀 중 몇 %가 바뀌면 움직임으로 볼지 (낮을수록 민감)
    force_ocr_sec: float = 1.2        # 배경이 계속 움직여도 이 간격마다 한 번은 읽음
    min_confidence: float = 40.0      # 이보다 낮은 줄은 잡음으로 보고 버림
    review_confidence: float = 80.0   # 이보다 낮으면 '확인 필요' 표시
    ellipsis: str = "……"              # 말줄임표 통일 형식 ("……", "...", "......", "" = 그대로)
    line_join: str = "auto"           # 줄바꿈 연결: auto / space / none
    replacements: List[List[str]] = field(default_factory=list)  # [["hill", "{PLAYER}"], ...] 결과 치환
    regions: List[Region] = field(default_factory=list)
    # 화면 표지: 읽을 때마다 이 자리의 밝기(0~255, 중앙값)를 잰다. 예: {"name": "창 밝기", "rect": [x0, y0, x1, y1]}
    marks: List[dict] = field(default_factory=list)
    # 형태 규칙: 대사창 결과의 형태를 위에서부터 처음 맞는 규칙으로 정한다. 조건은 모두 생략 가능.
    #   {"kind": "주인공속말", "speaker": false, "text_left_min": 0.28, "mark": "창 밝기", "mark_max": 120}
    #   from: 영역이 정한 기본 형태가 이것일 때만 · speaker: 이름표가 있어야(true) / 없어야(false)
    #   text_left_min/max: 본문 첫 글자의 가로 위치(비율)
    #   mark + mark_min/mark_max: 화면 표지의 밝기 범위
    kind_rules: List[dict] = field(default_factory=list)
    messenger_self: str = "나"        # 메신저에서 색 말풍선(내가 보낸 것)의 화자 이름
    messenger_self_tone: float = 60.0  # 줄 바탕 채도가 이보다 높으면 색 말풍선
    choice_marks: List[str] = field(default_factory=list)  # 카드 화면의 상태 표시 글(카드 글에서 빼고 메모로)
    ui_phrases: List[str] = field(default_factory=list)  # 이 글이 보이면 게임 UI 화면(대사 아님) — 예: "스토리 입장"
    choice_question: str = ""  # 카드 화면의 질문 말풍선 글이 맞아야 할 정규식 (예: "\\?$" — 물음표로 끝남)
    notes: str = ""

    # ---- 조회 ----
    @property
    def areas(self) -> List[Region]:
        return [r for r in self.regions if r.type == AREA]

    @property
    def zones(self) -> List[Region]:
        return [r for r in self.regions if r.type in ZONE_TYPES]

    # ---- 저장/불러오기 ----
    def to_dict(self) -> dict:
        d = asdict(self)
        d["regions"] = [
            {k: v for k, v in asdict(r).items() if v not in ("", None) or k in ("type", "rect")}
            for r in self.regions
        ]
        for r in d["regions"]:
            r["rect"] = [round(float(v), 5) for v in r["rect"]]
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        known = {k for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kw = {k: v for k, v in d.items() if k in known and k != "regions"}
        p = cls(**kw)
        p.regions = []
        for r in d.get("regions", []):
            if r.get("type") not in (AREA,) + ZONE_TYPES:
                continue
            rect = tuple(float(v) for v in r["rect"])
            if len(rect) != 4:
                continue
            x0, y0, x1, y1 = rect
            rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            hd = r.get("header")
            p.regions.append(Region(type=r["type"], rect=rect, label=r.get("label", ""),
                                    label_no_speaker=r.get("label_no_speaker", ""), mode=r.get("mode", ""),
                                    header=tuple(float(v) for v in hd) if hd and len(hd) == 4 else None))
        return p

    @classmethod
    def load(cls, path: str) -> "Profile":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def validate(self) -> Optional[str]:
        """문제가 있으면 사용자에게 보여줄 메시지를 돌려준다."""
        if not self.areas:
            return "텍스트 영역이 하나도 없습니다. 영역 편집에서 '텍스트 영역'을 하나 이상 그려 주세요."
        for r in self.regions:
            x0, y0, x1, y1 = r.rect
            if x1 - x0 < 0.005 or y1 - y0 < 0.005:
                return f"너무 작은 영역이 있습니다 ({TYPE_NAMES.get(r.type, r.type)})."
        if not (1 <= self.sample_fps <= 30):
            return "초당 검사 프레임은 1~30 사이여야 합니다."
        if self.line_join not in ("auto", "space", "none"):
            return "줄바꿈 연결 방식은 auto/space/none 중 하나여야 합니다."
        return None


def list_profiles(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json"))

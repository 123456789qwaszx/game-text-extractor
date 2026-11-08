"""영상 1개 → 대사 목록 + 엑셀. GUI와 명령줄이 함께 쓰는 핵심 흐름."""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import video
from .ocr import Line, LineOCR, find_dot_line
from .postprocess import Reading, Record, build_records
from .profile import IGNORE, SPEAKER, Profile, Region

Box = Tuple[int, int, int, int]
ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    pass


@dataclass
class Result:
    records: List[Record]
    readings: int
    xlsx_path: str
    shots_dir: str
    duration: float
    elapsed: float


def _inter(a: Box, b: Box) -> Optional[Box]:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


class AreaWatcher:
    """영역 하나의 화면 변화를 지켜보다가 '읽을 순간'을 알려준다."""

    # 글자 한 자가 새로 찍혀도 알아챌 수 있도록 '평균 밝기 차이'가 아니라
    # '눈에 띄게 바뀐 픽셀의 개수'로 변화를 판단한다.
    SIG_W = 640       # 비교용 축소 이미지 폭 (1080p 기준 글자 한 자 ≈ 15px)
    PIX_DIFF = 28     # 이만큼 밝기가 바뀐 픽셀을 '바뀐 픽셀'로 센다

    def __init__(self, idx: int, box: Box, ignore: Sequence[Box], profile: Profile):
        self.idx, self.box = idx, box
        w, h = box[2] - box[0], box[3] - box[1]
        sw = min(self.SIG_W, w)
        self.sig_size = (sw, max(8, int(round(sw * h / max(1, w)))))
        sx, sy = self.sig_size[0] / w, self.sig_size[1] / h
        self.mask = np.ones(self.sig_size[::-1], bool)
        self.ignore_local: List[Box] = []
        for g in ignore:
            r = _inter(g, box)
            if not r:
                continue
            lx0, ly0, lx1, ly1 = r[0] - box[0], r[1] - box[1], r[2] - box[0], r[3] - box[1]
            self.ignore_local.append((lx0, ly0, lx1, ly1))
            self.mask[int(ly0 * sy):int(math.ceil(ly1 * sy)), int(lx0 * sx):int(math.ceil(lx1 * sx))] = False
        n = int(self.mask.sum())
        # change_threshold: 영역 픽셀 중 몇 %가 바뀌어야 '화면이 움직였다'고 볼지
        self.min_px = max(8, int(n * profile.change_threshold / 100.0))
        # 화면 전환(대사창이 뜨고 사라짐, 장면 변경)으로 볼 큰 변화의 기준
        self.trans_px = max(self.min_px * 10, int(n * 0.05))
        # 배경 캐릭터가 계속 움직여 화면이 멈추지 않을 때: 0.3초마다 '글자 위치'만 빠르게 확인
        self.probe_every = 0.25
        self.last_probe = -1e9
        self.last_fp: Optional[list] = None   # 마지막으로 읽었을 때의 글자 줄 위치
        self.probe_fp: Optional[list] = None  # 직전 확인 때의 글자 줄 위치
        self.probe_rows: Optional[list] = None
        self.probe_t = 0.0
        self.probe_frame: Optional[np.ndarray] = None
        self.need = max(1, int(round(profile.stable_sec * profile.sample_fps)))
        self.force = profile.force_ocr_sec
        self.prev: Optional[np.ndarray] = None
        self.read_sig: Optional[np.ndarray] = None  # 마지막으로 읽었을 때의 화면
        self.still = 0
        self.fired = False
        self.moving_since: Optional[float] = None
        self.last_ocr = -1e9
        self.recheck_at: Optional[float] = None  # 결과가 불확실하면 잠시 후 한 번 더 읽는다

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return frame[y0:y1, x0:x1]

    def _changed(self, a: np.ndarray, b: np.ndarray) -> int:
        d = cv2.absdiff(a, b) > self.PIX_DIFF
        return int(np.count_nonzero(d & self.mask))

    def update(self, t: float, crop: np.ndarray) -> Optional[str]:
        """읽을 때가 되면 이유를 돌려준다.
        'stable': 화면이 멈춤 / 'exit': 잠깐 멈췄다가 바로 넘어감 → 직전 화면을 읽음
        'probe': 배경이 움직이는 중 → 글자 위치가 바뀌었으면 읽음
        'forced': 큰 변화가 계속됨 → 일정 간격으로 읽음"""
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sig = cv2.resize(g, self.sig_size, interpolation=cv2.INTER_AREA)
        prev = self.prev
        changed = self._changed(sig, prev) if prev is not None else 10 ** 9
        moving = changed > self.min_px
        self.prev = sig
        if moving:
            was_still, was_fired = self.still, self.fired
            self.still = 0
            self.fired = False
            self.recheck_at = None
            if self.moving_since is None:
                self.moving_since = t
            # 대사가 짧게 떴다가 바로 넘어간 경우: 멈춰 있던 직전 화면을 읽는다
            if was_still >= 1 and not was_fired and prev is not None and (
                    self.read_sig is None or self._changed(prev, self.read_sig) > self.min_px):
                self.read_sig = prev
                self.last_ocr = t
                return "exit"
            # 배경 애니메이션 정도의 변화가 이어지면(전환은 아님) 글자 위치가 바뀌었는지 확인한다
            if changed <= self.trans_px and t - self.last_probe >= self.probe_every:
                self.last_probe = t
                return "probe"
            # 큰 화면 전환이 시작됨: 확인만 해 둔 글자가 있으면 사라지기 전에 읽는다
            if changed > self.trans_px and self.probe_fp:
                return "flush"
        else:
            self.still += 1
        if self.still >= self.need and not self.fired:
            self.fired = True
            self.moving_since = None
            # 마지막으로 (멈춘 상태에서) 읽은 화면과 같으면 다시 읽지 않는다
            if self.read_sig is not None and self._changed(sig, self.read_sig) <= self.min_px:
                return None
            self.read_sig = sig
            self.last_ocr = t
            return "stable"
        if not moving and self.recheck_at is not None and t >= self.recheck_at:
            self.recheck_at = None
            self.last_ocr = t
            return "stable"
        if (self.still == 0 and self.moving_since is not None
                and t - self.moving_since >= self.force and t - self.last_ocr >= self.force):
            self.last_ocr = t  # 강제로 읽은 화면은 기준 화면으로 삼지 않는다 (전환 중일 수 있음)
            return "forced"
        return None

    def set_probe(self, fp, rows, t, frame) -> None:
        self.probe_fp, self.probe_rows, self.probe_t, self.probe_frame = fp, rows, t, frame

    def clear_probe(self) -> None:
        self.probe_fp, self.probe_rows, self.probe_frame = None, None, None

    @staticmethod
    def grows(a: list, b: list, tol: int = 12) -> bool:
        """b가 a에서 글자가 이어서 늘어난 상태인지 (타이핑 진행 중)."""
        if not a or not b or len(b) < len(a):
            return False
        for ra, rb in zip(a, b):
            if abs(ra[0] - rb[0]) > tol or abs(ra[1] - rb[1]) > tol or rb[2] < ra[2] - tol:
                return False
        return True

    @staticmethod
    def fingerprint(rows) -> list:
        return [tuple(int(v) for v in r) for r in rows]

    @staticmethod
    def same_fp(a: Optional[list], b: Optional[list], tol: int = 12) -> bool:
        if a is None or b is None or len(a) != len(b):
            return False
        return all(all(abs(x - y) <= tol for x, y in zip(ra, rb)) for ra, rb in zip(a, b))

    def feedback(self, t: float, stable: bool, weak: bool) -> None:
        """멈춘 화면을 읽었는데 결과가 불확실하면(본문 없음/신뢰도 낮음) 0.5초 뒤 한 번 더 읽는다."""
        if stable and weak and self.recheck_at is None:
            self.recheck_at = t + 0.5
        elif not weak:
            self.recheck_at = None


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def extract(video_path: str, profile: Profile, out_dir: Optional[str] = None,
            progress: Optional[ProgressFn] = None, cancel: Optional[Callable[[], bool]] = None,
            ocr: Optional[LineOCR] = None, tesseract_cmd: Optional[str] = None,
            debug: bool = False) -> Result:
    from .export_xlsx import write_xlsx

    err = profile.validate()
    if err:
        raise ValueError(err)
    started = time.time()
    progress = progress or (lambda f, m: None)
    cancel = cancel or (lambda: False)

    info = video.probe(video_path)
    W, H = info.width, info.height
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = out_dir or os.path.dirname(os.path.abspath(video_path))
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = _unique(os.path.join(out_dir, f"{stem}_텍스트.xlsx"))
    shots_dir = os.path.splitext(xlsx_path)[0] + "_스크린샷"
    os.makedirs(shots_dir, exist_ok=True)
    for f in os.listdir(shots_dir):  # 이전 실행에서 남은 스크린샷 정리
        if re.match(r"^(\d{4}_.*|_r\d+)\.jpg$", f):
            try:
                os.remove(os.path.join(shots_dir, f))
            except OSError:
                pass

    progress(0.0, "글자 인식 엔진 준비 중...")
    ocr = ocr or LineOCR(lang=profile.language, tesseract_cmd=tesseract_cmd)

    zones_px = [(z, z.to_px(W, H)) for z in profile.zones]
    speaker_px = [b for z, b in zones_px if z.type == SPEAKER]

    def in_speaker_zone(ln: Line, abox: Box) -> bool:
        ax = ln.box[0] + 0.3 * ln.h + abox[0]
        ay = (ln.box[1] + ln.box[3]) / 2 + abox[1]
        return any(b[0] <= ax <= b[2] and b[1] <= ay <= b[3] for b in speaker_px)

    ignore_px = [b for z, b in zones_px if z.type == IGNORE]
    watchers = [AreaWatcher(i, a.to_px(W, H), ignore_px, profile) for i, a in enumerate(profile.areas)]

    readings: List[Reading] = []
    state = {"rid": 0}
    shot_for: dict = {}

    def do_read(wt: AreaWatcher, why: str, rt: float, rframe: np.ndarray, rows) -> None:
        crop = wt.crop(rframe)
        wt.last_fp = wt.fingerprint(rows)
        lines = ocr.read(crop, min_conf=profile.min_confidence, ellipsis=profile.ellipsis,
                         ignore=wt.ignore_local, rows=rows)
        lines = [ln for ln in lines if _keep(ln)]
        if len(lines) == 1 and in_speaker_zone(lines[0], wt.box):
            dots = find_dot_line(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), lines[0], profile.ellipsis)
            if dots is not None:
                lines.append(dots)
        weak = len(lines) < 2 or min(l.conf for l in lines) < profile.review_confidence
        wt.feedback(rt, why == "stable", weak and bool(lines))
        if not lines:
            return
        if rt not in shot_for:
            shot_for[rt] = os.path.join(shots_dir, f"_r{state['rid']:05d}.jpg")
            _imwrite(shot_for[rt], rframe)
        ox, oy = wt.box[0], wt.box[1]
        glines = [Line(l.text, l.conf, (l.box[0] + ox, l.box[1] + oy, l.box[2] + ox, l.box[3] + oy))
                  for l in lines]
        readings.append(Reading(state["rid"], rt, wt.idx, wt.box, glines, why != "forced", shot_for[rt]))
        state["rid"] += 1

    last_report = 0.0
    dur = max(info.duration, 0.001)
    prev_t, prev_frame = 0.0, None
    try:
        for t, frame in video.iter_samples(video_path, profile.sample_fps):
            if cancel():
                raise Cancelled()
            shot_for.clear()
            for wt in watchers:
                why = wt.update(t, wt.crop(frame))
                if why is None:
                    continue
                if why == "flush":
                    # 배경 애니메이션 중 확인만 해 두고 아직 읽지 않은 글자가 화면 전환으로 사라지려 함 → 지금 읽는다
                    if wt.probe_rows and not wt.same_fp(wt.probe_fp, wt.last_fp):
                        do_read(wt, "probe", wt.probe_t, wt.probe_frame, wt.probe_rows)
                    wt.clear_probe()
                    continue
                rt, rframe = (prev_t, prev_frame) if (why == "exit" and prev_frame is not None) else (t, frame)
                rows = ocr.detect_rows(wt.crop(rframe), ignore=wt.ignore_local)
                fp = wt.fingerprint(rows)
                if why == "probe":
                    prev_fp = wt.probe_fp
                    if wt.same_fp(fp, prev_fp):
                        # 글자 위치가 직전 확인 때와 같음 = 타이핑이 끝나 멈춤 → 아직 안 읽었으면 읽는다
                        if rows and not wt.same_fp(fp, wt.last_fp):
                            do_read(wt, "probe", rt, rframe, rows)
                    elif (prev_fp and wt.probe_rows and not wt.same_fp(prev_fp, wt.last_fp)
                          and not wt.grows(prev_fp, fp)):
                        # 글자가 이어서 늘어난 게 아니라 다른 글로 바뀜 → 직전 확인 화면을 놓치지 않고 읽는다
                        do_read(wt, "probe", wt.probe_t, wt.probe_frame, wt.probe_rows)
                    wt.set_probe(fp, rows, rt, rframe)
                    continue
                do_read(wt, why, rt, rframe, rows)
                wt.set_probe(fp, rows, rt, rframe)
            prev_t, prev_frame = t, frame
            now = time.time()
            if now - last_report > 0.3:
                last_report = now
                progress(min(0.97, t / dur), f"분석 중 {fmt_time(t)} / {fmt_time(dur)}  (읽은 장면 {len(readings)}개)")
    except BaseException:
        _cleanup_temp(shots_dir)  # 중지/오류 시 임시 스크린샷 정리
        raise

    progress(0.97, "대사 정리 중...")
    records = build_records(readings, profile, zones_px)
    if debug:  # 문제 분석용: 읽은 순간 전체를 JSON으로 남긴다
        import json
        with open(os.path.join(shots_dir, "_debug_readings.json"), "w", encoding="utf-8") as f:
            json.dump([{"t": round(r.t, 2), "stable": r.stable, "area": r.area_idx,
                        "lines": [[l.text, round(l.conf, 1), list(l.box)] for l in r.lines]}
                       for r in readings], f, ensure_ascii=False, indent=1)

    # 사용된 스크린샷만 남기고 이름 정리
    used = {}
    for i, r in enumerate(records, 1):
        if r.shot and os.path.exists(r.shot):
            new = os.path.join(shots_dir, f"{i:04d}_{int(r.t // 60):02d}m{r.t % 60:05.2f}s.jpg")
            if r.shot in used:
                r.shot = used[r.shot]
                continue
            try:
                os.replace(r.shot, new)
                used[r.shot] = new
                r.shot = new
            except OSError:
                pass
    for f in os.listdir(shots_dir):
        if f.startswith("_r") and f.endswith(".jpg"):
            try:
                os.remove(os.path.join(shots_dir, f))
            except OSError:
                pass

    progress(0.98, "엑셀 파일 만드는 중...")
    write_xlsx(xlsx_path, records, profile=profile, video_info=info, shots_dir=shots_dir,
               engine=f"PP-OCRv5 검출 + {ocr.recognizer.version()}")
    elapsed = time.time() - started
    progress(1.0, f"완료: {len(records)}줄 추출 ({elapsed:.0f}초)")
    return Result(records, len(readings), xlsx_path, shots_dir, info.duration, elapsed)


def _cleanup_temp(shots_dir: str) -> None:
    try:
        for f in os.listdir(shots_dir):
            if f.startswith("_r") and f.endswith(".jpg"):
                os.remove(os.path.join(shots_dir, f))
        if not os.listdir(shots_dir):
            os.rmdir(shots_dir)
    except OSError:
        pass


def _keep(ln: Line) -> bool:
    # 한 글자짜리 잡음(아이콘, 배경 무늬)을 거른다
    core = ln.text.replace(" ", "")
    if len(core) <= 1 and ln.conf < 85:
        return False
    return True


def _imwrite(path: str, frame: np.ndarray) -> None:
    # cv2.imwrite는 Windows 한글 경로에서 실패하므로 인코딩 후 직접 쓴다
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def _unique(path: str) -> str:
    """같은 이름의 파일이 열려 있거나 이미 있으면 (2), (3)... 을 붙인다."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for i in range(2, 1000):
        p = f"{base} ({i}){ext}"
        if not os.path.exists(p):
            return p
    return path

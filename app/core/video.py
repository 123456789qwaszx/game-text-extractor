"""영상 읽기 (PyAV). 한글 경로, 가변 프레임레이트(VFR) 녹화도 문제없이 처리한다."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    duration: float  # 초
    fps: float


def _open(path: str):
    import av

    if not os.path.isfile(path):
        raise VideoError(f"영상 파일을 찾을 수 없습니다: {path}")
    try:
        c = av.open(path)
    except Exception as e:  # noqa: BLE001
        raise VideoError(f"영상을 열 수 없습니다: {e}") from e
    if not c.streams.video:
        c.close()
        raise VideoError("영상 트랙이 없는 파일입니다.")
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    return c, s


def probe(path: str) -> VideoInfo:
    c, s = _open(path)
    try:
        dur = 0.0
        if s.duration and s.time_base:
            dur = float(s.duration * s.time_base)
        elif c.duration:
            dur = c.duration / 1_000_000
        fps = float(s.average_rate) if s.average_rate else 30.0
        w, h = s.codec_context.width, s.codec_context.height
        return VideoInfo(path, w, h, dur, fps)
    finally:
        c.close()


def iter_samples(path: str, sample_fps: float, start: float = 0.0,
                 end: Optional[float] = None) -> Iterator[Tuple[float, np.ndarray]]:
    """sample_fps 간격으로 (시간, BGR 프레임)을 돌려준다."""
    c, s = _open(path)
    step = 1.0 / sample_fps
    nxt = start
    t0 = None
    try:
        if start > 0:
            c.seek(int(start / s.time_base), stream=s, backward=True, any_frame=False)
        for frame in c.decode(s):
            t = frame.time
            if t is None:
                continue
            if t0 is None:
                t0 = float(s.start_time * s.time_base) if s.start_time else 0.0
            t = t - t0
            if end is not None and t > end:
                break
            if t + 1e-6 < nxt:
                continue
            yield t, frame.to_ndarray(format="bgr24")
            nxt += step
            while nxt <= t:
                nxt += step
    finally:
        c.close()


def frame_at(path: str, t: float) -> Tuple[float, np.ndarray]:
    """t초 지점의 프레임 한 장 (영역 편집 화면용)."""
    c, s = _open(path)
    try:
        t0 = float(s.start_time * s.time_base) if s.start_time else 0.0
        c.seek(int((t + t0) / s.time_base), stream=s, backward=True, any_frame=False)
        last = None
        for frame in c.decode(s):
            if frame.time is None:
                continue
            last = frame
            if frame.time - t0 >= t - 1e-3:
                break
        if last is None:
            raise VideoError("프레임을 읽을 수 없습니다.")
        return float(last.time - t0), last.to_ndarray(format="bgr24")
    finally:
        c.close()

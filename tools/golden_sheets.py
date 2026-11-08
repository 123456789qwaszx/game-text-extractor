"""정답표 초안용 — segments.json의 구간마다 (화면 전체 축소판 | 대사창 원래 해상도)를 붙여 읽기 판을 만든다.

python tools/golden_sheets.py <영상> <segments 폴더> <출력 폴더> [--per 6]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np


def grab_all(video: str, times, work: str):
    """영상을 한 번만 처음부터 풀며 times 근처의 프레임들을 고른다(구간마다 탐색하면 느리다).
    고른 프레임의 실제 시각은 showinfo가 알려 준 pts_time을 쓴다 — 가변 프레임레이트에서도 어긋나지 않는다.
    → [(시각, 파일)]"""
    import re
    os.makedirs(work, exist_ok=True)
    for n in os.listdir(work):
        os.remove(os.path.join(work, n))
    expr = "+".join(f"between(t,{t - 0.04:.3f},{t + 0.0:.3f})" for t in times)
    with open(os.path.join(work, "select.txt"), "w") as fp:
        fp.write(f"select='{expr}',showinfo")
    r = subprocess.run(["ffmpeg", "-v", "info", "-y", "-i", video, "-filter_script:v", os.path.join(work, "select.txt"),
                        "-vsync", "0", os.path.join(work, "%06d.png")], capture_output=True, text=True, check=True)
    stamps = [float(m) for m in re.findall(r"\] n:\s*\d+ .*?pts_time:([0-9.]+)", r.stderr)]
    files = sorted(n for n in os.listdir(work) if n.endswith(".png"))
    if len(stamps) != len(files):
        raise RuntimeError(f"프레임 {len(files)}장과 시각 {len(stamps)}개가 맞지 않는다")
    return list(zip(stamps, files))


def _time_base(video: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base",
                        "-of", "csv=p=0", video], capture_output=True, text=True, check=True)
    n, d = r.stdout.strip().split("/")
    return float(n) / float(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("segments")
    ap.add_argument("out")
    ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.segments, "segments.json"), encoding="utf-8"))
    x0, y0, x1, y1 = meta["area"]
    os.makedirs(a.out, exist_ok=True)
    tiles = []
    work = os.path.join(a.out, "_frames")
    times = [max(0.0, s["t1"] - 0.05) for s in meta["segments"]]      # 구간 끝(글자가 다 나온 상태)
    by_t = grab_all(a.video, times, work)
    for s, want in zip(meta["segments"], times):
        got_t, name = min(by_t, key=lambda x: abs(x[0] - want))
        if abs(got_t - want) > 0.1:
            raise RuntimeError(f"#{s['id']}: {want:.2f}초 근처 프레임이 없다(가장 가까운 것 {got_t:.2f}초)")
        f = cv2.imread(os.path.join(work, name))
        H, W = f.shape[:2]
        area = f[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        area = cv2.resize(area, (1100, int(area.shape[0] * 1100 / area.shape[1])))
        thumb = cv2.resize(f, (int(W * area.shape[0] / H), area.shape[0]))
        tile = np.hstack([thumb, np.full((area.shape[0], 6, 3), 255, np.uint8), area])
        cv2.rectangle(tile, (0, 0), (260, 34), (0, 0, 0), -1)
        cv2.putText(tile, f"#{s['id']}  {s['t0']:.1f}-{s['t1']:.1f}s", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)
        tiles.append(tile)
    w = max(t.shape[1] for t in tiles)
    tiles = [np.hstack([t, np.full((t.shape[0], w - t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    for i in range(0, len(tiles), a.per):
        sheet = np.vstack(sum(([t, np.full((8, w, 3), 80, np.uint8)] for t in tiles[i:i + a.per]), []))
        cv2.imwrite(os.path.join(a.out, f"sheet_{i // a.per + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"판 {(len(tiles) + a.per - 1) // a.per}장 → {a.out}")


if __name__ == "__main__":
    main()

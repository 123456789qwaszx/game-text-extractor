"""정답표 초안용 — 영상을 '화면이 멈춘 구간'으로 나눈다. 추출기(app/)와 독립이다.

추출기가 빠뜨린 줄을 찾으려면 추출기 결과가 아니라 영상에서 출발해야 한다.
프레임을 일정 간격으로 받아, 대사창 영역과 화면 전체의 변화를 따로 잰다.
  - 멈춘 구간: 대사창 영역이 hold초 이상 거의 바뀌지 않은 구간(타자기가 끝난 뒤)
  - 앞 구간과 대사창이 같으면(같은 대사 위에서 배경만 움직임) 하나로 합친다
구간마다 마지막 프레임(글자가 다 나온 상태)의 대사창 · 화면 전체를 PNG로 남기고 segments.json을 쓴다.

python tools/golden_segments.py <영상> <출력 폴더> [--area x0 y0 x1 y1] [--fps 10] [--hold 0.4] [--until 초]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np

PIX = 28          # 이만큼 밝기가 바뀐 화소를 '바뀐 화소'로 센다(추출기 AreaWatcher와 같은 값)


def frames(path: str, fps: float, width: int, until: float | None):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    w, h = (int(v) for v in probe.stdout.strip().split(","))
    oh = int(round(h * width / w / 2)) * 2
    cmd = ["ffmpeg", "-v", "error"] + (["-t", str(until)] if until else []) + [
        "-i", path, "-vf", f"fps={fps},scale={width}:{oh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n, size = 0, width * oh * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield n / fps, np.frombuffer(buf, np.uint8).reshape(oh, width, 3)
        n += 1
    p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--area", type=float, nargs=4, default=[0.17923, 0.68242, 0.82663, 0.97462])
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--hold", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--until", type=float)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    segs, cur = [], None
    prev_area = None
    for t, f in frames(a.video, a.fps, a.width, a.until):
        H, W = f.shape[:2]
        x0, y0, x1, y1 = int(a.area[0] * W), int(a.area[1] * H), int(a.area[2] * W), int(a.area[3] * H)
        area = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        moved = prev_area is None or np.count_nonzero(cv2.absdiff(area, prev_area) > PIX) > 0.002 * area.size
        prev_area = area
        if moved:
            if cur and cur["t1"] - cur["t0"] >= a.hold:
                segs.append(cur)
            cur = {"t0": t, "t1": t, "frame": f, "area": area}
        else:
            cur["t1"], cur["frame"], cur["area"] = t, f, area
    if cur and cur["t1"] - cur["t0"] >= a.hold:
        segs.append(cur)

    merged = []
    for s in segs:      # 같은 대사창이 이어지면(가운데 화면만 움직였다) 합친다
        if merged and np.count_nonzero(cv2.absdiff(s["area"], merged[-1]["area"]) > PIX) <= 0.002 * s["area"].size:
            merged[-1]["t1"] = s["t1"]
            merged[-1]["frame"] = s["frame"]
            continue
        merged.append(s)

    out = []
    for i, s in enumerate(merged, 1):
        name = f"{i:04d}_{s['t0']:07.2f}"
        cv2.imwrite(os.path.join(a.out, name + ".png"), s["frame"])
        out.append({"id": i, "t0": round(s["t0"], 2), "t1": round(s["t1"], 2), "image": name + ".png"})
    with open(os.path.join(a.out, "segments.json"), "w", encoding="utf-8") as fp:
        json.dump({"video": os.path.basename(a.video), "fps": a.fps, "hold": a.hold, "area": a.area,
                   "segments": out}, fp, ensure_ascii=False, indent=1)
    print(f"구간 {len(out)}개 (멈춘 구간 {len(segs)}개를 합침) → {a.out}")


if __name__ == "__main__":
    main()

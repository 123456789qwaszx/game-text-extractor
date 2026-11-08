"""번역 작업용 엑셀 만들기.

시트 구성
- 텍스트: No / 시간 / 형태 / 화자 / 원문 / 번역(빈칸) / 메모 / 확인 / 신뢰도 / 스크린샷 / 원본
- 화자 목록: 이름 번역(용어집)용
- 정보: 영상·프로필·추출 조건
"""
from __future__ import annotations

import datetime as _dt
import io
import math
import os
from collections import Counter
from typing import List, Optional, Sequence

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

THUMB_MAX_W = 620
THUMB_TEXT_H = 19  # 썸네일 속 글자 높이(px) 목표

HEAD_FILL = PatternFill("solid", fgColor="305496")
HEAD_FONT = Font(bold=True, color="FFFFFF")
INPUT_FILL = PatternFill("solid", fgColor="E2EFDA")
REVIEW_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")
CENTER_TOP = Alignment(horizontal="center", vertical="top")


def _fmt(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _read_image(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)  # 한글 경로 대응
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def _thumb(rec) -> Optional[tuple]:
    img = _read_image(rec.shot) if rec.shot else None
    if img is None:
        return None
    H, W = img.shape[:2]
    if rec.lines:
        x0 = min(l.box[0] for l in rec.lines)
        y0 = min(l.box[1] for l in rec.lines)
        x1 = max(l.box[2] for l in rec.lines)
        y1 = max(l.box[3] for l in rec.lines)
        lh = float(np.median([l.box[3] - l.box[1] for l in rec.lines]))
    else:
        x0, y0, x1, y1 = rec.area_box
        lh = 40.0
    m = int(0.6 * lh)
    ax0, ay0, ax1, ay1 = rec.area_box
    x0, y0 = max(ax0, x0 - m, 0), max(ay0, y0 - m, 0)
    x1, y1 = min(ax1, x1 + m, W), min(ay1, y1 + m, H)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    scale = min(THUMB_TEXT_H / max(lh, 1), THUMB_MAX_W / crop.shape[1], 1.0)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return io.BytesIO(buf.tobytes()), crop.shape[1], crop.shape[0]


def _est_lines(text: str, width_chars: float) -> int:
    n = 0
    for part in (text or "").split("\n"):
        w = sum(2 if ord(ch) > 0x2E80 else 1 for ch in part)
        n += max(1, math.ceil(w / max(1.0, width_chars)))
    return n


def write_xlsx(path: str, records: Sequence, profile=None, video_info=None, shots_dir: str = "",
               engine: str = "") -> str:
    wb = Workbook()
    ws = wb.active
    ws.title = "텍스트"
    cols = [("No", 6), ("시간", 10), ("형태", 11), ("화자", 13), ("원문", 52), ("번역", 52),
            ("메모", 22), ("확인", 18), ("신뢰도", 8), ("스크린샷", 90), ("원본", 8)]
    for i, (name, w) in enumerate(cols, 1):
        c = ws.cell(row=1, column=i, value=name)
        c.fill, c.font, c.border = HEAD_FILL, HEAD_FONT, BORDER
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.cell(row=1, column=6).fill = PatternFill("solid", fgColor="548235")
    ws.row_dimensions[1].height = 22
    ws.freeze_panes = "E2"

    base = os.path.dirname(os.path.abspath(path))
    for n, r in enumerate(records, 1):
        row = n + 1
        vals = [n, _fmt(r.t), r.kind, r.speaker, r.text, "", "", r.review, int(round(r.conf))]
        for ci, v in enumerate(vals, 1):
            c = ws.cell(row=row, column=ci, value=v)
            c.border = BORDER
            c.alignment = CENTER_TOP if ci in (1, 2, 3, 9) else WRAP_TOP
        ws.cell(row=row, column=6).fill = INPUT_FILL
        if r.needs_review:
            ws.cell(row=row, column=5).fill = REVIEW_FILL
            ws.cell(row=row, column=8).font = Font(color="C00000")
        elif r.review:
            ws.cell(row=row, column=8).font = Font(color="7F7F7F")
        for ci in (10, 11):
            ws.cell(row=row, column=ci).border = BORDER

        text_h = 15 * _est_lines(r.text, 50) + 6
        img_h = 0
        th = _thumb(r)
        if th:
            bio, tw, tht = th
            im = XLImage(bio)
            im.width, im.height = tw, tht
            ws.add_image(im, f"J{row}")
            img_h = tht * 0.75 + 6
        ws.row_dimensions[row].height = max(20, text_h, img_h)
        if r.shot and os.path.exists(r.shot):
            c = ws.cell(row=row, column=11, value="열기")
            c.hyperlink = os.path.relpath(r.shot, base)
            c.font = Font(color="0563C1", underline="single")
            c.alignment = CENTER_TOP

    if records:
        ws.auto_filter.ref = f"A1:I{len(records) + 1}"
    # 인쇄/PDF 저장 시 한 페이지 폭에 맞춤
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_title_rows = "1:1"

    # ---- 화자 목록
    ws2 = wb.create_sheet("화자 목록")
    for i, (name, w) in enumerate([("화자(원문)", 18), ("대사 수", 9), ("첫 등장", 10), ("번역명", 22), ("메모", 30)], 1):
        c = ws2.cell(row=1, column=i, value=name)
        c.fill, c.font, c.border = HEAD_FILL, HEAD_FONT, BORDER
        ws2.column_dimensions[get_column_letter(i)].width = w
    cnt = Counter(r.speaker for r in records if r.speaker)
    first = {}
    for r in records:
        if r.speaker and r.speaker not in first:
            first[r.speaker] = r.t
    for row, (spk, k) in enumerate(sorted(cnt.items(), key=lambda kv: first[kv[0]]), 2):
        for ci, v in enumerate([spk, k, _fmt(first[spk]), "", ""], 1):
            c = ws2.cell(row=row, column=ci, value=v)
            c.border = BORDER
        ws2.cell(row=row, column=4).fill = INPUT_FILL
    ws2.freeze_panes = "A2"

    # ---- 정보
    ws3 = wb.create_sheet("정보")
    review = sum(1 for r in records if r.needs_review)
    info_rows = [
        ("추출 일시", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("영상 파일", os.path.basename(video_info.path) if video_info else ""),
        ("영상 길이", _fmt(video_info.duration) if video_info else ""),
        ("해상도", f"{video_info.width}x{video_info.height}" if video_info else ""),
        ("게임 프로필", profile.name if profile else ""),
        ("인식 언어", profile.language if profile else ""),
        ("추출된 줄 수", len(records)),
        ("확인 필요", review),
        ("스크린샷 폴더", os.path.basename(shots_dir) if shots_dir else ""),
        ("인식 엔진", engine),
        ("", ""),
        ("사용 안내", "노란색 원문 칸은 인식이 불확실한 줄입니다. 스크린샷과 대조해 원문을 고친 뒤 번역해 주세요."),
        ("", "'원본 열기'를 누르면 해당 장면의 전체 화면 스크린샷이 열립니다 (엑셀과 스크린샷 폴더를 같은 위치에 두세요)."),
    ]
    for row, (k, v) in enumerate(info_rows, 1):
        ws3.cell(row=row, column=1, value=k).font = Font(bold=True)
        ws3.cell(row=row, column=2, value=v)
    ws3.column_dimensions["A"].width = 16
    ws3.column_dimensions["B"].width = 100

    tmp = path + ".tmp.xlsx"
    wb.save(tmp)
    os.replace(tmp, path)
    return path

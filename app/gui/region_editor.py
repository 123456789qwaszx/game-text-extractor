"""영역 편집기: 영상 화면 위에 마우스로 대사창·화자 이름·무시 영역을 그려 게임 프로필을 만든다."""
from __future__ import annotations

import os
import re
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..core import paths, video
from ..core.postprocess import assign, clean_speaker
from ..core.profile import AREA, IGNORE, LABEL, SPEAKER, TYPE_NAMES, Profile, Region

COLORS = {AREA: "#2E9E44", SPEAKER: "#2F6FDB", LABEL: "#E07B00", IGNORE: "#8A8A8A"}
TYPE_HELP = {
    AREA: "글자를 읽을 전체 범위 (대사창, 나레이션, 장소 카드가 나오는 곳을 모두 포함)",
    SPEAKER: "화자 이름이 나오는 자리 (이름표). 여러 개 그려도 됩니다",
    LABEL: "대사와 따로 분류할 자리 (예: 장소/시간 카드, 시스템 메시지)",
    IGNORE: "읽지 않을 자리 (깜빡이는 '다음' 아이콘, 고정 버튼 글자 등)",
}
ELLIPSIS_CHOICES = ["……", "...", "......", "…", "(그대로)"]


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")
    return name or "새_프로필"


class RegionEditor(tk.Toplevel):
    MAX_W, MAX_H = 1180, 664

    def __init__(self, master, video_path: str, profile: Profile, profile_path: Optional[str],
                 on_saved: Callable[[str], None], get_ocr: Callable[[str], object]):
        super().__init__(master)
        self.title(f"영역 편집 - {profile.name}")
        self.video_path = video_path
        self.profile = Profile.from_dict(profile.to_dict())
        self.profile_path = profile_path
        self.on_saved = on_saved
        self.get_ocr = get_ocr
        self.info = video.probe(video_path)
        self.frame: Optional[np.ndarray] = None
        self.photo = None
        self.scale = 1.0
        self.drag_start = None
        self.drag_rect = None
        self.test_items: List[int] = []
        self.selected: Optional[int] = None
        self._build()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1560, sw - 40)}x{min(900, sh - 80)}+10+10")
        self.minsize(1100, 680)
        self.after(50, lambda: self.load_frame(min(self.info.duration * 0.25, 10.0)))
        self.transient(master)
        self.grab_set()

    # ------------------------------------------------------------------ UI
    def _build(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(root, width=340)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        # 시간 이동
        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="화면 위치").pack(side="left")
        self.time_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(bar, from_=0, to=max(1.0, self.info.duration - 0.1), variable=self.time_var,
                                orient="horizontal")
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.slider.bind("<ButtonRelease-1>", lambda e: self.load_frame(self.time_var.get()))
        for txt, d in (("◀ 5초", -5), ("◀ 0.5초", -0.5), ("0.5초 ▶", 0.5), ("5초 ▶", 5)):
            ttk.Button(bar, text=txt, width=7, command=lambda d=d: self.load_frame(self.time_var.get() + d)).pack(side="left", padx=1)
        self.time_label = ttk.Label(bar, text="00:00.0", width=9)
        self.time_label.pack(side="left", padx=(6, 0))

        self.canvas = tk.Canvas(left, width=self.MAX_W, height=self.MAX_H, bg="#202020", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>", lambda e: self._render())
        ttk.Label(left, text="대사가 화면에 다 나온 장면으로 이동한 뒤, 오른쪽에서 종류를 고르고 화면 위를 드래그해 영역을 그리세요.",
                  foreground="#555").pack(anchor="w", pady=(6, 0))

        # 오른쪽 패널: 위쪽은 탭(영역 그리기 / 인식 설정), 아래쪽은 항상 보이는 테스트·저장
        bottom = ttk.LabelFrame(right, text="확인 및 저장", padding=6)
        bottom.pack(side="bottom", fill="x", pady=(8, 0))
        nb = ttk.Notebook(right)
        nb.pack(side="top", fill="both", expand=True)
        tab1 = ttk.Frame(nb, padding=6)
        tab2 = ttk.Frame(nb, padding=6)
        nb.add(tab1, text="영역 그리기")
        nb.add(tab2, text="인식 설정")

        box = ttk.LabelFrame(tab1, text="1. 그릴 영역 종류", padding=6)
        box.pack(fill="x")
        self.type_var = tk.StringVar(value=AREA)
        for t in (AREA, SPEAKER, LABEL, IGNORE):
            rb = ttk.Radiobutton(box, text=TYPE_NAMES[t], value=t, variable=self.type_var, command=self._on_type)
            rb.pack(anchor="w")
        self.type_help = ttk.Label(box, text=TYPE_HELP[AREA], wraplength=310, foreground="#555")
        self.type_help.pack(anchor="w", pady=(4, 0))
        self.label_frame = ttk.Frame(box)
        self.label_frame.pack(fill="x", pady=(6, 0))
        self.label_var = tk.StringVar(value="대사")
        self.label2_var = tk.StringVar(value="나레이션")
        self.label_lbl = ttk.Label(self.label_frame, text="화자가 있을 때 형태 이름")
        self.label_ent = ttk.Entry(self.label_frame, textvariable=self.label_var)
        self.label2_lbl = ttk.Label(self.label_frame, text="화자가 없을 때 형태 이름")
        self.label2_ent = ttk.Entry(self.label_frame, textvariable=self.label2_var)
        self._on_type()

        box2 = ttk.LabelFrame(tab1, text="2. 그린 영역 (클릭하면 화면에 굵게 표시)", padding=6)
        box2.pack(fill="both", expand=True, pady=(8, 0))
        self.tree = ttk.Treeview(box2, columns=("type", "label"), show="headings", height=6)
        self.tree.heading("type", text="종류")
        self.tree.heading("label", text="이름")
        self.tree.column("type", width=100)
        self.tree.column("label", width=180)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        bb = ttk.Frame(box2)
        bb.pack(fill="x", pady=(4, 0))
        ttk.Button(bb, text="선택 삭제", command=self._delete_selected).pack(side="left")
        ttk.Button(bb, text="모두 지우기", command=self._clear_all).pack(side="left", padx=4)

        box3 = ttk.Frame(tab2)
        box3.pack(fill="x")
        grid = ttk.Frame(box3)
        grid.pack(fill="x")
        langs = paths.available_languages() or ["kor"]
        self.lang_var = tk.StringVar(value=self.profile.language)
        ttk.Label(grid, text="언어").grid(row=0, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.lang_var, values=sorted(set(langs + ["kor+eng"])), width=12).grid(row=0, column=1, sticky="w")
        self.ell_var = tk.StringVar(value=self.profile.ellipsis or "(그대로)")
        ttk.Label(grid, text="말줄임표").grid(row=1, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.ell_var, values=ELLIPSIS_CHOICES, width=12, state="readonly").grid(row=1, column=1, sticky="w")
        self.join_var = tk.StringVar(value=self.profile.line_join)
        ttk.Label(grid, text="줄바꿈 연결").grid(row=2, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.join_var, values=["auto", "space", "none"], width=12,
                     state="readonly").grid(row=2, column=1, sticky="w")
        self.fps_var = tk.DoubleVar(value=self.profile.sample_fps)
        ttk.Label(grid, text="초당 검사 횟수").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(grid, from_=2, to=30, increment=1, textvariable=self.fps_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(box3, text="결과 바꾸기 (한 줄에 하나)\n예) hill => {PLAYER}   플레이어 이름을 변수로\n예) 취황의 칼날 => 취황의 칼날   고유명사 교정",
                  foreground="#555").pack(anchor="w", pady=(10, 2))
        self.rep_text = tk.Text(box3, height=6, width=36)
        self.rep_text.pack(fill="x")
        self.rep_text.insert("1.0", "\n".join(f"{a} => {b}" for a, b in self.profile.replacements))

        ttk.Label(box3, text="· 언어: 한국어 게임은 kor (영문 이름이 섞이면 자동으로 영어도 함께 읽음)\n"
                              "· 줄바꿈 연결 auto: 자동 줄바꿈된 줄을 띄어쓰기까지 판단해 이어 붙임\n"
                              "· 초당 검사 횟수: 대사를 빨리 넘기는 영상이면 높게(10~15)",
                  foreground="#555", wraplength=310, justify="left").pack(anchor="w", pady=(10, 0))

        box4 = bottom
        ttk.Button(box4, text="이 화면으로 인식 테스트", command=self.run_test).pack(fill="x")
        self.result = tk.Text(box4, height=5, width=36, wrap="word")
        self.result.pack(fill="x", pady=(4, 6))
        nm = ttk.Frame(box4)
        nm.pack(fill="x")
        ttk.Label(nm, text="프로필 이름").pack(side="left")
        self.name_var = tk.StringVar(value=self.profile.name)
        ttk.Entry(nm, textvariable=self.name_var).pack(side="left", fill="x", expand=True, padx=(6, 0))
        sv = ttk.Frame(box4)
        sv.pack(fill="x", pady=(6, 0))
        ttk.Button(sv, text="저장", command=self.save).pack(side="left", fill="x", expand=True)
        ttk.Button(sv, text="닫기", command=self.destroy).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._refresh_tree()

    def _on_type(self):
        t = self.type_var.get()
        self.type_help.config(text=TYPE_HELP[t])
        for w in (self.label_lbl, self.label_ent, self.label2_lbl, self.label2_ent):
            w.pack_forget()
        if t == AREA:
            self.label_lbl.config(text="화자가 있을 때 형태 이름")
            self.label_var.set(self.label_var.get() if self.label_var.get() not in ("장소/시간",) else "대사")
            for w in (self.label_lbl, self.label_ent, self.label2_lbl, self.label2_ent):
                w.pack(fill="x")
        elif t == LABEL:
            self.label_lbl.config(text="분류 이름 (엑셀 '형태' 칸에 표시)")
            if self.label_var.get() in ("대사", ""):
                self.label_var.set("장소/시간")
            self.label_lbl.pack(fill="x")
            self.label_ent.pack(fill="x")

    # ------------------------------------------------------------------ 화면
    def load_frame(self, t: float):
        t = max(0.0, min(t, max(0.0, self.info.duration - 0.05)))
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            real_t, frame = video.frame_at(self.video_path, t)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("영상 오류", str(e), parent=self)
            return
        finally:
            self.config(cursor="")
        self.frame = frame
        self.time_var.set(real_t)
        m, s = divmod(real_t, 60)
        self.time_label.config(text=f"{int(m):02d}:{s:04.1f}")
        self._clear_test()
        self._render()

    def _render(self):
        if self.frame is None:
            return
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        h, w = self.frame.shape[:2]
        self.scale = min(cw / w, ch / h)
        nw, nh = int(w * self.scale), int(h * self.scale)
        self.ox, self.oy = (cw - nw) // 2, (ch - nh) // 2
        img = cv2.cvtColor(cv2.resize(self.frame, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete("all")
        self.canvas.create_image(self.ox, self.oy, image=self.photo, anchor="nw")
        self._draw_regions()

    def _to_rel(self, x, y):
        h, w = self.frame.shape[:2]
        rx = (x - self.ox) / (w * self.scale)
        ry = (y - self.oy) / (h * self.scale)
        return min(1.0, max(0.0, rx)), min(1.0, max(0.0, ry))

    def _to_canvas(self, rx, ry):
        h, w = self.frame.shape[:2]
        return self.ox + rx * w * self.scale, self.oy + ry * h * self.scale

    def _draw_regions(self):
        self.canvas.delete("region")
        for i, r in enumerate(self.profile.regions):
            x0, y0 = self._to_canvas(r.rect[0], r.rect[1])
            x1, y1 = self._to_canvas(r.rect[2], r.rect[3])
            col = COLORS.get(r.type, "#fff")
            width = 4 if i == self.selected else 2
            dash = (4, 3) if r.type == IGNORE else None
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=width, dash=dash, tags="region")
            name = TYPE_NAMES[r.type] + (f": {r.label}" if r.type == LABEL and r.label else "")
            self.canvas.create_text(x0 + 4, y0 + 2, text=name, anchor="nw", fill=col,
                                    font=("Malgun Gothic", 10, "bold"), tags="region")

    # ------------------------------------------------------------------ 그리기
    def _on_press(self, e):
        if self.frame is None:
            return
        self.drag_start = (e.x, e.y)
        col = COLORS[self.type_var.get()]
        self.drag_rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline=col, width=2, dash=(5, 3))

    def _on_drag(self, e):
        if self.drag_rect:
            x0, y0 = self.drag_start
            self.canvas.coords(self.drag_rect, x0, y0, e.x, e.y)

    def _on_release(self, e):
        if not self.drag_rect:
            return
        x0, y0 = self.drag_start
        self.canvas.delete(self.drag_rect)
        self.drag_rect = None
        if abs(e.x - x0) < 6 or abs(e.y - y0) < 6:
            return
        a, b = self._to_rel(min(x0, e.x), min(y0, e.y))
        c, d = self._to_rel(max(x0, e.x), max(y0, e.y))
        t = self.type_var.get()
        label = self.label_var.get().strip() if t in (AREA, LABEL) else ""
        label2 = self.label2_var.get().strip() if t == AREA else ""
        self.profile.regions.append(Region(type=t, rect=(a, b, c, d), label=label, label_no_speaker=label2))
        self.selected = len(self.profile.regions) - 1
        self._refresh_tree()
        self._draw_regions()

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.profile.regions):
            lab = r.label if r.type in (LABEL,) else (f"{r.label} / {r.label_no_speaker}" if r.type == AREA else "")
            self.tree.insert("", "end", iid=str(i), values=(TYPE_NAMES[r.type], lab))
        if self.selected is not None and self.selected < len(self.profile.regions):
            self.tree.selection_set(str(self.selected))

    def _on_select(self, _e=None):
        sel = self.tree.selection()
        self.selected = int(sel[0]) if sel else None
        self._draw_regions()

    def _delete_selected(self):
        if self.selected is None:
            return
        del self.profile.regions[self.selected]
        self.selected = None
        self._refresh_tree()
        self._draw_regions()

    def _clear_all(self):
        if self.profile.regions and messagebox.askyesno("확인", "그린 영역을 모두 지울까요?", parent=self):
            self.profile.regions.clear()
            self.selected = None
            self._refresh_tree()
            self._draw_regions()

    # ------------------------------------------------------------------ 테스트/저장
    def _collect(self) -> Profile:
        p = self.profile
        p.name = self.name_var.get().strip() or "새 프로필"
        p.language = self.lang_var.get().strip() or "kor"
        e = self.ell_var.get()
        p.ellipsis = "" if e == "(그대로)" else e
        p.line_join = self.join_var.get()
        try:
            p.sample_fps = float(self.fps_var.get())
        except (tk.TclError, ValueError):
            p.sample_fps = 10.0
        reps = []
        for ln in self.rep_text.get("1.0", "end").splitlines():
            if "=>" in ln:
                a, b = ln.split("=>", 1)
                if a.strip():
                    reps.append([a.strip(), b.strip()])
        p.replacements = reps
        return p

    def _clear_test(self):
        for it in self.test_items:
            self.canvas.delete(it)
        self.test_items = []

    def run_test(self):
        p = self._collect()
        err = p.validate()
        if err:
            messagebox.showwarning("확인", err, parent=self)
            return
        self.config(cursor="watch")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", "인식하는 중...")
        frame = self.frame.copy()

        def work():
            try:
                ocr = self.get_ocr(p.language)
                h, w = frame.shape[:2]
                zones = [(z, z.to_px(w, h)) for z in p.zones]
                ign = [b for z, b in zones if z.type == IGNORE]
                out = []
                for a in p.areas:
                    ab = a.to_px(w, h)
                    crop = frame[ab[1]:ab[3], ab[0]:ab[2]]
                    local_ign = [(max(g[0], ab[0]) - ab[0], max(g[1], ab[1]) - ab[1],
                                  min(g[2], ab[2]) - ab[0], min(g[3], ab[3]) - ab[1]) for g in ign
                                 if min(g[2], ab[2]) > max(g[0], ab[0]) and min(g[3], ab[3]) > max(g[1], ab[1])]
                    lines = ocr.read(crop, min_conf=p.min_confidence, ellipsis=p.ellipsis, ignore=local_ign)
                    from ..core.ocr import Line
                    glines = [Line(l.text, l.conf, (l.box[0] + ab[0], l.box[1] + ab[1], l.box[2] + ab[0], l.box[3] + ab[1]))
                              for l in lines]
                    spk, labels, body = assign(glines, zones)
                    out.append((glines, spk, labels, body))
                self.after(0, lambda: self._show_test(out))
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                self.after(0, lambda: self._test_error(msg))

        threading.Thread(target=work, daemon=True).start()

    def _test_error(self, msg):
        self.config(cursor="")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", "오류: " + msg)

    def _show_test(self, out):
        self.config(cursor="")
        self._clear_test()
        self.result.delete("1.0", "end")
        texts = []
        for glines, spk, labels, body in out:
            for ln, col in [(l, COLORS[SPEAKER]) for l in spk] + [(l, COLORS[AREA]) for l in body] + \
                           [(l, COLORS[LABEL]) for v in labels.values() for l in v]:
                x0, y0 = self._to_canvas(ln.box[0] / self.frame.shape[1], ln.box[1] / self.frame.shape[0])
                x1, y1 = self._to_canvas(ln.box[2] / self.frame.shape[1], ln.box[3] / self.frame.shape[0])
                self.test_items.append(self.canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=2))
            if spk:
                texts.append("[화자] " + clean_speaker(" ".join(l.text for l in spk)))
            for l in body:
                texts.append(f"[본문] {l.text}  ({l.conf:.0f})")
            for name, v in labels.items():
                for l in v:
                    texts.append(f"[{name}] {l.text}")
        self.result.insert("1.0", "\n".join(texts) or "읽힌 글자가 없습니다. 영역 위치를 확인하세요.")

    def save(self):
        p = self._collect()
        err = p.validate()
        if err:
            messagebox.showwarning("확인", err, parent=self)
            return
        os.makedirs(paths.PROFILES_DIR, exist_ok=True)
        path = os.path.join(paths.PROFILES_DIR, safe_filename(p.name) + ".json")
        if os.path.exists(path) and path != self.profile_path:
            if not messagebox.askyesno("덮어쓰기", f"'{p.name}' 프로필이 이미 있습니다. 덮어쓸까요?", parent=self):
                return
        p.save(path)
        self.profile_path = path
        self.on_saved(path)
        messagebox.showinfo("저장 완료", f"프로필을 저장했습니다.\n{path}", parent=self)

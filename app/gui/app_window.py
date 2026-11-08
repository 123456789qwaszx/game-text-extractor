"""메인 창: 영상 선택 → 게임 프로필 선택 → 추출 시작 → 엑셀 열기."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

from ..core import paths
from ..core.ocr import LineOCR, OcrError
from ..core.pipeline import Cancelled, extract
from ..core.profile import Profile, list_profiles
from .region_editor import RegionEditor

APP_TITLE = "게임 영상 텍스트 추출기"
VIDEO_TYPES = [("영상 파일", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.flv"), ("모든 파일", "*.*")]


def open_path(path: str) -> None:
    try:
        if paths.IS_WINDOWS:
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as e:
        messagebox.showerror("열기 실패", str(e))


class MainWindow(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master = master
        self.pack(fill="both", expand=True)
        self.q: "queue.Queue" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.cancel_flag = threading.Event()
        self.ocr_cache: Dict[str, LineOCR] = {}
        self.last_xlsx: Optional[str] = None
        self.settings = paths.load_settings()
        self._build()
        self.reload_profiles(self.settings.get("last_profile"))
        self.after(100, self._poll)
        self.after(300, self.check_tesseract)

    # ------------------------------------------------------------------ 화면 구성
    def _build(self):
        head = ttk.Label(self, text=APP_TITLE, font=("Malgun Gothic", 15, "bold"))
        head.pack(anchor="w")
        ttk.Label(self, text="게임 플레이 녹화 영상에서 대사·나레이션을 읽어 번역 작업용 엑셀로 저장합니다.",
                  foreground="#555").pack(anchor="w", pady=(0, 10))

        f1 = ttk.LabelFrame(self, text="① 영상 파일 (여러 개 선택 가능)", padding=8)
        f1.pack(fill="x")
        lf = ttk.Frame(f1)
        lf.pack(side="left", fill="both", expand=True)
        self.video_list = tk.Listbox(lf, height=4, selectmode="extended")
        self.video_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.video_list.yview)
        sb.pack(side="left", fill="y")
        self.video_list.config(yscrollcommand=sb.set)
        bf = ttk.Frame(f1)
        bf.pack(side="left", padx=(8, 0), fill="y")
        ttk.Button(bf, text="영상 추가", command=self.add_videos).pack(fill="x")
        ttk.Button(bf, text="선택 제거", command=self.remove_videos).pack(fill="x", pady=4)
        ttk.Button(bf, text="모두 비우기", command=lambda: self.video_list.delete(0, "end")).pack(fill="x")

        f2 = ttk.LabelFrame(self, text="② 게임 프로필 (대사창 위치 설정)", padding=8)
        f2.pack(fill="x", pady=(10, 0))
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(f2, textvariable=self.profile_var, state="readonly", width=48)
        self.profile_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(f2, text="영역 편집 / 테스트", command=self.edit_profile).pack(side="left", padx=(8, 0))
        ttk.Button(f2, text="새 프로필 만들기", command=self.new_profile).pack(side="left", padx=(4, 0))

        f3 = ttk.LabelFrame(self, text="③ 저장 위치", padding=8)
        f3.pack(fill="x", pady=(10, 0))
        self.out_var = tk.StringVar(value=self.settings.get("out_dir", ""))
        ttk.Entry(f3, textvariable=self.out_var).pack(side="left", fill="x", expand=True)
        ttk.Button(f3, text="폴더 선택", command=self.pick_out).pack(side="left", padx=(8, 0))
        ttk.Label(f3, text="(비워 두면 영상과 같은 폴더)", foreground="#666").pack(side="left", padx=(8, 0))

        f4 = ttk.Frame(self)
        f4.pack(fill="x", pady=(12, 0))
        self.start_btn = ttk.Button(f4, text="▶ 추출 시작", command=self.start)
        self.start_btn.pack(side="left", ipadx=16, ipady=4)
        self.stop_btn = ttk.Button(f4, text="■ 중지", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0), ipady=4)
        self.open_btn = ttk.Button(f4, text="결과 엑셀 열기", command=self.open_result, state="disabled")
        self.open_btn.pack(side="right", ipady=4)
        self.folder_btn = ttk.Button(f4, text="결과 폴더 열기", command=self.open_folder, state="disabled")
        self.folder_btn.pack(side="right", padx=(0, 6), ipady=4)

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", pady=(10, 2))
        self.status = ttk.Label(self, text="대기 중")
        self.status.pack(anchor="w")

        lf2 = ttk.LabelFrame(self, text="작업 기록", padding=4)
        lf2.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(lf2, height=10, wrap="word", state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        sb2 = ttk.Scrollbar(lf2, orient="vertical", command=self.log.yview)
        sb2.pack(side="left", fill="y")
        self.log.config(yscrollcommand=sb2.set)

        menu = tk.Menu(self.master)
        m1 = tk.Menu(menu, tearoff=0)
        m1.add_command(label="Tesseract 위치 지정...", command=self.pick_tesseract)
        m1.add_command(label="프로필 폴더 열기", command=lambda: open_path(paths.PROFILES_DIR))
        menu.add_cascade(label="설정", menu=m1)
        m2 = tk.Menu(menu, tearoff=0)
        m2.add_command(label="사용 설명서", command=self.open_manual)
        menu.add_cascade(label="도움말", menu=m2)
        self.master.config(menu=menu)

    # ------------------------------------------------------------------ 로그
    def write_log(self, msg: str):
        self.log.config(state="normal")
        self.log.insert("end", time.strftime("[%H:%M:%S] ") + msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ------------------------------------------------------------------ 입력
    def add_videos(self):
        files = filedialog.askopenfilenames(title="영상 선택", filetypes=VIDEO_TYPES,
                                            initialdir=self.settings.get("video_dir") or None)
        existing = set(self.video_list.get(0, "end"))
        for f in files:
            if f not in existing:
                self.video_list.insert("end", f)
        if files:
            self.settings["video_dir"] = os.path.dirname(files[0])
            paths.save_settings({"video_dir": self.settings["video_dir"]})

    def remove_videos(self):
        for i in reversed(self.video_list.curselection()):
            self.video_list.delete(i)

    def pick_out(self):
        d = filedialog.askdirectory(title="저장 폴더 선택")
        if d:
            self.out_var.set(d)

    def reload_profiles(self, select_path: Optional[str] = None):
        self.profile_paths = list_profiles(paths.PROFILES_DIR)
        names = []
        for p in self.profile_paths:
            try:
                names.append(Profile.load(p).name)
            except Exception:  # noqa: BLE001
                names.append(os.path.basename(p) + " (읽기 오류)")
        self.profile_combo["values"] = names
        idx = 0
        if select_path and select_path in self.profile_paths:
            idx = self.profile_paths.index(select_path)
        if names:
            self.profile_combo.current(idx)

    def current_profile_path(self) -> Optional[str]:
        i = self.profile_combo.current()
        if i is None or i < 0 or i >= len(self.profile_paths):
            return None
        return self.profile_paths[i]

    def _first_video(self) -> Optional[str]:
        sel = self.video_list.curselection()
        items = self.video_list.get(0, "end")
        if sel:
            return items[sel[0]]
        return items[0] if items else None

    def get_ocr(self, lang: str) -> LineOCR:
        if lang not in self.ocr_cache:
            self.ocr_cache[lang] = LineOCR(lang=lang)
        return self.ocr_cache[lang]

    def edit_profile(self):
        v = self._first_video()
        if not v:
            messagebox.showinfo("영상 필요", "영역을 그리려면 먼저 ①에서 영상을 추가해 주세요.")
            return
        p = self.current_profile_path()
        prof = Profile.load(p) if p else Profile()
        RegionEditor(self.master, v, prof, p, on_saved=self._on_profile_saved, get_ocr=self.get_ocr)

    def new_profile(self):
        v = self._first_video()
        if not v:
            messagebox.showinfo("영상 필요", "영역을 그리려면 먼저 ①에서 영상을 추가해 주세요.")
            return
        RegionEditor(self.master, v, Profile(name="새 게임"), None, on_saved=self._on_profile_saved,
                     get_ocr=self.get_ocr)

    def _on_profile_saved(self, path: str):
        self.reload_profiles(path)
        self.write_log(f"프로필 저장: {os.path.basename(path)}")

    # ------------------------------------------------------------------ 설정
    def check_tesseract(self):
        if paths.find_tesseract():
            return
        if messagebox.askyesno("Tesseract 필요",
                               "글자 인식 엔진(Tesseract)을 찾지 못했습니다.\n"
                               "설치.bat을 실행했다면 tesseract.exe 위치를 직접 지정할 수 있습니다.\n\n지금 지정할까요?"):
            self.pick_tesseract()

    def pick_tesseract(self):
        f = filedialog.askopenfilename(title="tesseract.exe 선택",
                                       filetypes=[("tesseract", "tesseract.exe tesseract"), ("모든 파일", "*.*")])
        if f:
            paths.save_settings({"tesseract_cmd": f})
            self.ocr_cache.clear()
            self.write_log(f"Tesseract 위치 저장: {f}")

    def open_manual(self):
        f = os.path.join(paths.APP_ROOT, "사용설명서.txt")
        if os.path.exists(f):
            open_path(f)

    # ------------------------------------------------------------------ 실행
    def start(self):
        videos = list(self.video_list.get(0, "end"))
        if not videos:
            messagebox.showinfo("영상 필요", "추출할 영상을 추가해 주세요.")
            return
        ppath = self.current_profile_path()
        if not ppath:
            messagebox.showinfo("프로필 필요", "게임 프로필을 선택하거나 새로 만들어 주세요.")
            return
        try:
            prof = Profile.load(ppath)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("프로필 오류", str(e))
            return
        err = prof.validate()
        if err:
            messagebox.showwarning("프로필 확인", err)
            return
        out = self.out_var.get().strip() or None
        paths.save_settings({"last_profile": ppath, "out_dir": out or ""})
        self.cancel_flag.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        self.write_log(f"시작: 영상 {len(videos)}개, 프로필 '{prof.name}'")
        self.worker = threading.Thread(target=self._run, args=(videos, prof, out), daemon=True)
        self.worker.start()

    def stop(self):
        self.cancel_flag.set()
        self.status.config(text="중지하는 중...")

    def _run(self, videos, prof, out):
        try:
            ocr = self.get_ocr(prof.language)
        except OcrError as e:
            self.q.put(("error", str(e)))
            self.q.put(("done", None))
            return
        n = len(videos)
        for i, v in enumerate(videos):
            if self.cancel_flag.is_set():
                break
            name = os.path.basename(v)
            self.q.put(("log", f"[{i + 1}/{n}] {name} 분석 시작"))

            def prog(f, msg, i=i):
                self.q.put(("progress", ((i + f) / n, f"[{i + 1}/{n}] {name} — {msg}")))
            try:
                res = extract(v, prof, out, progress=prog, cancel=self.cancel_flag.is_set, ocr=ocr)
                review = sum(1 for r in res.records if r.needs_review)
                self.q.put(("log", f"[{i + 1}/{n}] 완료: {len(res.records)}줄 (확인 필요 {review}줄), "
                                   f"{res.elapsed:.0f}초 → {res.xlsx_path}"))
                self.q.put(("result", res.xlsx_path))
            except Cancelled:
                self.q.put(("log", "사용자가 중지했습니다."))
                break
            except Exception as e:  # noqa: BLE001
                self.q.put(("log", f"[{i + 1}/{n}] 오류: {e}"))
                self.q.put(("log", traceback.format_exc(limit=3)))
        self.q.put(("done", None))

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "progress":
                    f, msg = val
                    self.progress["value"] = int(f * 1000)
                    self.status.config(text=msg)
                elif kind == "log":
                    self.write_log(val)
                elif kind == "error":
                    messagebox.showerror("오류", val)
                elif kind == "result":
                    self.last_xlsx = val
                    self.open_btn.config(state="normal")
                    self.folder_btn.config(state="normal")
                elif kind == "done":
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if not self.cancel_flag.is_set() and self.last_xlsx:
                        self.progress["value"] = 1000
                        self.status.config(text="완료")
                    else:
                        self.status.config(text="중지됨")
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def open_result(self):
        if self.last_xlsx and os.path.exists(self.last_xlsx):
            open_path(self.last_xlsx)

    def open_folder(self):
        if self.last_xlsx:
            open_path(os.path.dirname(self.last_xlsx))

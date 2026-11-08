"""프로그램 시작점 (GUI)."""
from __future__ import annotations

import os
import sys
import traceback


def _setup_dpi():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    # pythonw로 실행하면 표준출력이 없어서, 라이브러리가 경고를 출력하다 멈추는 것을 막는다
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    _setup_dpi()
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("게임 영상 텍스트 추출기")
    root.geometry("980x720")
    root.minsize(820, 600)
    families = set(tkfont.families())
    for fam in ("Malgun Gothic", "맑은 고딕", "Noto Sans CJK KR", "NanumGothic"):
        if fam in families:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                tkfont.nametofont(name).configure(family=fam, size=10)
            break
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")

    def report(exc, val, tb):
        msg = "".join(traceback.format_exception(exc, val, tb))
        try:
            log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "error.log")
            with open(log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass
        messagebox.showerror("오류", f"{val}\n\n자세한 내용은 error.log에 저장했습니다.")

    root.report_callback_exception = report
    from .gui.app_window import MainWindow
    MainWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

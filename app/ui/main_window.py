"""Desktop GUI — customtkinter preferred, tkinter fallback."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk

from app.analysis.engine import AnalysisResult, analyze_image

try:
    import customtkinter as ctk

    USE_CTK = True
except ImportError:
    import tkinter as ctk  # type: ignore
    from tkinter import ttk

    USE_CTK = False

import tkinter as tk
from tkinter import filedialog, messagebox


APP_TITLE = "图像取证助手 — 数字加工检测"
SUPPORTED = [("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("全部", "*.*")]


def _score_bar(score: float) -> str:
    filled = int(round(score * 10))
    return "█" * filled + "░" * (10 - filled) + f"  {score:.0%}"


class MainWindow:
    def __init__(self) -> None:
        if USE_CTK:
            ctk.set_appearance_mode("System")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self._photo_orig: Optional[ImageTk.PhotoImage] = None
        self._photo_ela: Optional[ImageTk.PhotoImage] = None
        self._current_path: Optional[Path] = None
        self._analyzing = False

        self._build()

    # --- widget helpers (CTK vs tk) ---
    def _frame(self, parent, **kw):
        if USE_CTK:
            return ctk.CTkFrame(parent, **kw)
        return tk.Frame(parent, **kw)

    def _label(self, parent, text="", **kw):
        if USE_CTK:
            return ctk.CTkLabel(parent, text=text, **kw)
        return tk.Label(parent, text=text, **kw)

    def _button(self, parent, text, command, **kw):
        if USE_CTK:
            return ctk.CTkButton(parent, text=text, command=command, **kw)
        return tk.Button(parent, text=text, command=command, **kw)

    def _textbox(self, parent, **kw):
        if USE_CTK:
            return ctk.CTkTextbox(parent, **kw)
        return tk.Text(parent, **kw)

    def _build(self) -> None:
        top = self._frame(self.root)
        top.pack(fill="x", padx=12, pady=8)

        self._button(top, text="打开图像…", command=self.open_image, width=120).pack(
            side="left", padx=(0, 8)
        )
        self.analyze_btn = self._button(
            top, text="开始分析", command=self.start_analyze, width=120
        )
        self.analyze_btn.pack(side="left", padx=(0, 8))
        self.path_label = self._label(top, text="未选择文件", anchor="w")
        self.path_label.pack(side="left", fill="x", expand=True, padx=8)

        # Verdict strip
        verdict_fr = self._frame(self.root)
        verdict_fr.pack(fill="x", padx=12, pady=4)
        self.verdict_label = self._label(
            verdict_fr,
            text="结论：—",
            font=("Microsoft YaHei UI", 18, "bold") if not USE_CTK else ("Microsoft YaHei UI", 20),
        )
        self.verdict_label.pack(side="left", padx=(4, 16))
        self.conf_label = self._label(verdict_fr, text="置信度：—")
        self.conf_label.pack(side="left", padx=8)
        self.risk_label = self._label(verdict_fr, text="风险分：—")
        self.risk_label.pack(side="left", padx=8)

        # Main body: images | details
        body = self._frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=8)

        left = self._frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = self._frame(body)
        right.pack(side="right", fill="both", expand=True, padx=(6, 0))

        self._label(left, text="原图").pack(anchor="w")
        self.canvas_orig = tk.Label(left, bg="#1a1a1a", text="原图预览", fg="#888")
        self.canvas_orig.pack(fill="both", expand=True, pady=4)

        self._label(left, text="ELA 热力图（差异放大显示）").pack(anchor="w")
        self.canvas_ela = tk.Label(left, bg="#1a1a1a", text="分析后显示 ELA", fg="#888")
        self.canvas_ela.pack(fill="both", expand=True, pady=4)

        self._label(right, text="检测详情").pack(anchor="w")
        self.detail_box = self._textbox(right, height=20)
        self.detail_box.pack(fill="both", expand=True, pady=4)
        self._set_detail_text(
            "使用说明：\n"
            "1. 点击「打开图像」选择 JPG/PNG 等文件\n"
            "2. 点击「开始分析」\n"
            "3. 查看结论、置信度与各项检查详情\n"
            "4. 左侧对比原图与 ELA 可视化\n\n"
            "本工具为启发式参考，非司法鉴定。"
        )

        # Disclaimer footer
        foot = self._label(
            self.root,
            text="免责声明：启发式检测仅供参考，不保证准确性，不能作为法律证据。",
            text_color="#888888" if USE_CTK else None,
        )
        if not USE_CTK:
            foot.configure(fg="#666666")
        foot.pack(fill="x", padx=12, pady=(0, 8))

        # Optional drag-drop (tkinterdnd2 not required)
        self.root.bind("<Control-o>", lambda e: self.open_image())

    def _set_detail_text(self, text: str) -> None:
        if USE_CTK and hasattr(self.detail_box, "delete"):
            self.detail_box.delete("1.0", "end")
            self.detail_box.insert("1.0", text)
        else:
            self.detail_box.delete("1.0", tk.END)
            self.detail_box.insert(tk.END, text)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="选择图像", filetypes=SUPPORTED)
        if not path:
            return
        self._current_path = Path(path)
        if USE_CTK:
            self.path_label.configure(text=str(self._current_path))
        else:
            self.path_label.config(text=str(self._current_path))
        self._show_preview_only(self._current_path)

    def _fit(self, im: Image.Image, max_w: int, max_h: int) -> Image.Image:
        im = im.copy()
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return im

    def _show_preview_only(self, path: Path) -> None:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                disp = self._fit(im, 480, 280)
                self._photo_orig = ImageTk.PhotoImage(disp)
                self.canvas_orig.configure(image=self._photo_orig, text="")
                self.canvas_ela.configure(image="", text="点击「开始分析」生成 ELA")
        except Exception as exc:
            messagebox.showerror("错误", f"无法打开图像：{exc}")

    def start_analyze(self) -> None:
        if self._analyzing:
            return
        if not self._current_path or not self._current_path.is_file():
            messagebox.showwarning("提示", "请先打开一张图像")
            return
        self._analyzing = True
        try:
            self.analyze_btn.configure(state="disabled", text="分析中…")
        except Exception:
            pass
        path = self._current_path
        threading.Thread(target=self._analyze_worker, args=(path,), daemon=True).start()

    def _analyze_worker(self, path: Path) -> None:
        try:
            result = analyze_image(path)
            self.root.after(0, lambda: self._apply_result(result))
        except Exception:
            err = traceback.format_exc()
            self.root.after(0, lambda: self._analyze_failed(err))

    def _analyze_failed(self, err: str) -> None:
        self._analyzing = False
        try:
            self.analyze_btn.configure(state="normal", text="开始分析")
        except Exception:
            pass
        messagebox.showerror("分析失败", err[:800])

    def _apply_result(self, result: AnalysisResult) -> None:
        self._analyzing = False
        try:
            self.analyze_btn.configure(state="normal", text="开始分析")
        except Exception:
            pass

        color_map = {
            "可能原图": "#2ecc71",
            "可能被加工": "#e74c3c",
            "不确定": "#f39c12",
        }
        color = color_map.get(result.verdict, "#ffffff")
        vtext = f"结论：{result.verdict}"
        if USE_CTK:
            self.verdict_label.configure(text=vtext, text_color=color)
            self.conf_label.configure(text=f"置信度：{result.confidence:.0%}")
            self.risk_label.configure(text=f"风险分：{result.overall_risk:.0%}")
        else:
            self.verdict_label.config(text=vtext, fg=color)
            self.conf_label.config(text=f"置信度：{result.confidence:.0%}")
            self.risk_label.config(text=f"风险分：{result.overall_risk:.0%}")

        # Images
        if result.original_preview is not None:
            disp = self._fit(result.original_preview.convert("RGB"), 480, 280)
            self._photo_orig = ImageTk.PhotoImage(disp)
            self.canvas_orig.configure(image=self._photo_orig, text="")
        if result.ela_image is not None:
            disp_e = self._fit(result.ela_image.convert("RGB"), 480, 280)
            self._photo_ela = ImageTk.PhotoImage(disp_e)
            self.canvas_ela.configure(image=self._photo_ela, text="")

        lines = [
            f"文件：{result.path}",
            f"结论：{result.verdict}　置信度：{result.confidence:.0%}　综合风险：{result.overall_risk:.0%}",
            "",
            "【简要说明】",
            result.explanation,
            "",
            "【元数据摘要】",
        ]
        for k, v in result.metadata_summary.items():
            lines.append(f"  · {k}: {v}")
        lines.append("")
        lines.append("【分项检查】")
        for chk in result.checks:
            lines.append(f"\n▸ {chk.name}  {_score_bar(chk.score)}")
            for f in chk.findings:
                lines.append(f"    - {f}")
            if chk.extras:
                extras = ", ".join(f"{k}={v}" for k, v in chk.extras.items())
                lines.append(f"    (细节: {extras})")
        lines.append("")
        lines.append("【免责声明】")
        lines.append(result.disclaimer)
        self._set_detail_text("\n".join(lines))

    def run(self) -> None:
        self.root.mainloop()


def run_app() -> None:
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    run_app()

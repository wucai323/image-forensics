"""Allow `python -m app` to launch the GUI."""

try:
    from app.ui.main_window import run_app
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        "无法启动图形界面：缺少依赖 "
        f"({missing})。\n"
        "请在 Windows 上安装 Python 官方安装包（勾选 tcl/tk），"
        "并执行: pip install -r requirements.txt\n"
        "无界面自检可运行: python scripts/self_check.py"
    ) from exc

if __name__ == "__main__":
    run_app()

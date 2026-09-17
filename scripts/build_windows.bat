@echo off
setlocal
cd /d "%~dp0.."
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller==6.10.0
pyinstaller --noconfirm --clean --windowed --onefile ^
  --name ImageForensics ^
  --collect-all customtkinter ^
  --hidden-import PIL._tkinter_finder ^
  main.py
echo.
echo Built: dist\ImageForensics.exe
endlocal

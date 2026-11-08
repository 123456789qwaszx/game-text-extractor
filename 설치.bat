@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title 게임 영상 텍스트 추출기 - 설치

echo ======================================================
echo   게임 영상 텍스트 추출기 설치
echo   (인터넷 연결 필요, 처음 한 번만 실행하면 됩니다)
echo ======================================================
echo.

rem ---------------------------------------------------------------- 1. Python
echo [1/4] Python 확인 중...
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PY (
        py -%%V -c "import tkinter" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys, tkinter; assert sys.version_info >= (3, 10)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo       Python이 없어 설치합니다. 잠시 기다려 주세요...
    winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
    )
)
if not defined PY (
    echo.
    echo  [오류] Python을 설치하지 못했습니다.
    echo         https://www.python.org/downloads/ 에서 Python 3.12를 설치한 뒤
    echo         이 파일을 다시 실행해 주세요. 설치할 때 "Add python.exe to PATH"를 체크하세요.
    goto :fail
)
%PY% --version

rem ---------------------------------------------------------------- 2. 프로그램 환경
echo.
echo [2/4] 프로그램 전용 환경 만드는 중... (처음에는 몇 분 걸립니다)
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo  [오류] 가상환경을 만들지 못했습니다.
        goto :fail
    )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo  [오류] 필요한 부품을 내려받지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.
    goto :fail
)

rem ---------------------------------------------------------------- 3. Tesseract
echo.
echo [3/4] 글자 인식 엔진(Tesseract) 확인 중...
call :find_tess
if not defined TESS (
    echo       Tesseract를 설치합니다. 관리자 권한 확인 창이 뜨면 '예'를 눌러 주세요.
    winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements
    call :find_tess
)
if not defined TESS (
    echo       winget 설치에 실패해 설치 파일을 직접 내려받습니다...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$o=$env:TEMP+'\tesseract_setup.exe'; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe' -OutFile $o"
    if exist "%TEMP%\tesseract_setup.exe" (
        "%TEMP%\tesseract_setup.exe" /S
        call :find_tess
    )
)
if not defined TESS (
    echo.
    echo  [오류] Tesseract를 설치하지 못했습니다.
    echo         https://github.com/UB-Mannheim/tesseract/wiki 에서 Windows용 설치 파일을 받아 설치한 뒤
    echo         이 파일을 다시 실행하거나, 프로그램의 [설정] - [Tesseract 위치 지정]에서 tesseract.exe를 지정하세요.
    goto :fail
)
echo       !TESS!
".venv\Scripts\python.exe" -c "from app.core import paths; paths.save_settings({'tesseract_cmd': r'!TESS!'})"

rem ---------------------------------------------------------------- 4. 확인
echo.
echo [4/4] 설치 확인 중...
".venv\Scripts\python.exe" -m app.selftest
if errorlevel 1 goto :fail

powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\게임 텍스트 추출기.lnk'); $s.TargetPath='%~dp0실행.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()" >nul 2>&1

echo.
echo ======================================================
echo   설치 완료. 바탕화면의 '게임 텍스트 추출기' 또는
echo   이 폴더의 '실행.bat'으로 프로그램을 시작하세요.
echo ======================================================
pause
exit /b 0

:find_tess
set "TESS="
for %%P in ("%ProgramFiles%\Tesseract-OCR\tesseract.exe" "%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe" "%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe" "%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe") do (
    if not defined TESS if exist "%%~P" set "TESS=%%~P"
)
if not defined TESS (
    for /f "delims=" %%i in ('where tesseract 2^>nul') do if not defined TESS set "TESS=%%i"
)
exit /b 0

:fail
echo.
echo 설치가 완료되지 않았습니다. 위의 안내를 확인해 주세요.
pause
exit /b 1

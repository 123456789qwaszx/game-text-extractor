@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 게임 영상 텍스트 추출기 (오류 확인용)
echo 프로그램이 바로 꺼지거나 이상할 때 이 파일로 실행하면 오류 내용을 볼 수 있습니다.
echo.
".venv\Scripts\python.exe" -m app.main
echo.
pause

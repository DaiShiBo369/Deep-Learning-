@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set XELATEX=C:\CTEX\MiKTeX\miktex\bin\x64\xelatex.exe
if not exist "%XELATEX%" (
  echo 找不到 xelatex: %XELATEX%
  echo 请确认 MiKTeX 安装位置，或直接找我改。
  pause
  exit /b 1
)
echo [1/3] 第一次编译...
"%XELATEX%" -interaction=nonstopmode HW2_report.tex >nul
echo [2/3] 第二次编译（生成目录/引用）...
"%XELATEX%" -interaction=nonstopmode HW2_report.tex >nul
if not exist HW2_report.pdf (
  echo 编译失败：没有生成 HW2_report.pdf，请查看 HW2_report.log
  pause
  exit /b 1
)
echo [3/3] 复制到提交文件夹...
copy /y HW2_report.pdf "..\..\戴士博2025012128_HW2\戴士博_2025012128_HW2_report.pdf" >nul
echo.
echo 完成：报告已重新生成并更新到提交文件夹。
echo 如需重新生成 zip，请再双击 HW2 目录下的 重新打包提交.bat。
pause

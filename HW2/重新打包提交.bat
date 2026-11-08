@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "戴士博2025012128_HW2\戴士博_2025012128_HW2_report.pdf" (
  echo 提交文件夹里没有报告 PDF，请先运行 work\report\重新生成报告.bat
  pause
  exit /b 1
)
echo 正在重新打包 戴士博2025012128_HW2.zip ...
powershell -NoProfile -Command "Compress-Archive -Path '戴士博2025012128_HW2' -DestinationPath '戴士博2025012128_HW2.zip' -Force"
if errorlevel 1 (
  echo 打包失败。
  pause
  exit /b 1
)
echo 完成：戴士博2025012128_HW2.zip 已更新。
pause

@echo off
title SHOT q1_3.png -- screenshot the output, then close
for /d %%D in ("%~dp0..\*") do if exist "%%D\hw2_q1_concept.py" cd /d "%%D"
"C:\Users\17857\anaconda3\envs\pytorch-learning\python.exe" hw2_q1_concept.py
echo.
echo ============================================================
echo  Save the output above as:  q1_3.png
echo  Save into folder: %~dp0
echo  Then close this window.
echo ============================================================
pause

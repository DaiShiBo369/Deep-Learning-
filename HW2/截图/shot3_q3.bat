@echo off
title SHOT Q3 -- q3_a / q3_b / q3_c
for /d %%D in ("%~dp0..\*") do if exist "%%D\hw2_q3_autograd.py" cd /d "%%D"
"C:\Users\17857\anaconda3\envs\pytorch-learning\python.exe" hw2_q3_autograd.py
echo.
echo ============================================================
echo  Take 2-3 screenshots (scroll with mouse wheel):
echo     q3_a.png  top: X, M and (a) first backward
echo     q3_b.png  (b) second backward: X.grad = 2M
echo     q3_c.png  (c) reset: X.grad = M again
echo  All three X.grad matrices must be readable.
echo  Save into folder: %~dp0
echo  Then close this window.
echo ============================================================
pause

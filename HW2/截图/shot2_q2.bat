@echo off
title SHOT Q2 -- q2_1 / q2_2 / q2_3
for /d %%D in ("%~dp0..\*") do if exist "%%D\hw2_q2_tensors.py" cd /d "%%D"
"C:\Users\17857\anaconda3\envs\pytorch-learning\python.exe" hw2_q2_tensors.py
echo.
echo ============================================================
echo  Take 3 screenshots (scroll with mouse wheel):
echo     q2_1.png  top: A, B, C and A+B, A-B
echo     q2_2.png  middle: ELEMENTWISE and MATRIX products
echo     q2_3.png  bottom: 2.2 shapes and indexing
echo  Save into folder: %~dp0
echo  Then close this window.
echo ============================================================
pause

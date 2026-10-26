@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\tangent_input_20261026\runs\explicit_small"
call "C:\SIMULIA\Commands\abaqus.bat" job=explicit_small input=explicit_small.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\elastic_dimensions_20261026\runs\CPS4_dx0p5_a34p5"
call "C:\SIMULIA\Commands\abaqus.bat" job=CPS4_dx0p5_a34p5 input=CPS4_dx0p5_a34p5.inp cpus=1 interactive > execution_stdout.txt 2>&1
exit /b %errorlevel%

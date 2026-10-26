@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\fixed_dcb_a34p5"
call "C:\SIMULIA\Commands\abaqus.bat" job=fixed_dcb_a34p5 input=fixed_dcb_a34p5.inp cpus=1 interactive > execution_stdout.txt 2>&1
exit /b %errorlevel%

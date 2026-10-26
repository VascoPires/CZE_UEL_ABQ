@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\fixed_as4_dcb"
call "C:\SIMULIA\Commands\abaqus.bat" job=fixed_as4_dcb input=fixed_as4_dcb.inp cpus=1 interactive > execution_stdout.txt 2>&1
exit /b %errorlevel%

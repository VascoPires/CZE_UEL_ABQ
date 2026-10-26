@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\dcb_native_local015"
call "C:\SIMULIA\Commands\abaqus.bat" job=dcb_native_local015 input=dcb_native_local015.inp cpus=1 interactive > execution_stdout.txt 2>&1
exit /b %errorlevel%

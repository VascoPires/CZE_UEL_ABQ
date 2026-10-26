@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\fixed_krueger2012_mmb80_open"
call "C:\SIMULIA\Commands\abaqus.bat" job=fixed_krueger2012_mmb80_open input=fixed_krueger2012_mmb80_open.inp cpus=1 interactive > execution_stdout.txt 2>&1
exit /b %errorlevel%

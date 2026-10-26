@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\dcb_adaptive_dx0p6"
call "C:\SIMULIA\Commands\abaqus.bat" job=dcb_adaptive_dx0p6 input=dcb_adaptive_dx0p6.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

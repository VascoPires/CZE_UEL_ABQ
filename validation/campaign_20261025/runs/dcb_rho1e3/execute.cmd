@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\dcb_rho1e3"
call "C:\SIMULIA\Commands\abaqus.bat" job=dcb_rho1e3 input=dcb_rho1e3.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\t2010_mmb50_open_adaptive"
call "C:\SIMULIA\Commands\abaqus.bat" job=t2010_mmb50_open_adaptive input=t2010_mmb50_open_adaptive.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

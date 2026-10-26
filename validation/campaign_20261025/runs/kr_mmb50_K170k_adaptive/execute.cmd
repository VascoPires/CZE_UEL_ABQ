@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\kr_mmb50_K170k_adaptive"
call "C:\SIMULIA\Commands\abaqus.bat" job=kr_mmb50_K170k_adaptive input=kr_mmb50_K170k_adaptive.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

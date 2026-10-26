@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\kr_mmb20_K170k"
call "C:\SIMULIA\Commands\abaqus.bat" job=kr_mmb20_K170k input=kr_mmb20_K170k.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

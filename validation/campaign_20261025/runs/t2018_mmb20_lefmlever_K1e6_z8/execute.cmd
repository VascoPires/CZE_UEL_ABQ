@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\t2018_mmb20_lefmlever_K1e6_z8"
call "C:\SIMULIA\Commands\abaqus.bat" job=t2018_mmb20_lefmlever_K1e6_z8 input=t2018_mmb20_lefmlever_K1e6_z8.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

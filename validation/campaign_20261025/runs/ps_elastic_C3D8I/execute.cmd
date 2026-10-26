@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\ps_elastic_C3D8I"
call "C:\SIMULIA\Commands\abaqus.bat" job=ps_elastic_C3D8I input=ps_elastic_C3D8I.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

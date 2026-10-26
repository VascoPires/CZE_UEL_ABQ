@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\contact_enf_elastic_K1e7"
call "C:\SIMULIA\Commands\abaqus.bat" job=contact_enf_elastic_K1e7 input=contact_enf_elastic_K1e7.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\elastic_turon2010_mmb50_n2s"
call "C:\SIMULIA\Commands\abaqus.bat" job=elastic_turon2010_mmb50_n2s input=elastic_turon2010_mmb50_n2s.inp cpus=1 interactive user=3D_CZM.f > execution_stdout.txt 2>&1
exit /b %errorlevel%

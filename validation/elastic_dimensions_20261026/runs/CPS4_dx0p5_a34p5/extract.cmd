@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\elastic_dimensions_20261026\runs\CPS4_dx0p5_a34p5"
call "C:\SIMULIA\Commands\abaqus.bat" python "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\elastic_dimensions_20261026\scripts\extract.py" > extraction_attempt_02.txt 2>&1
exit /b %errorlevel%

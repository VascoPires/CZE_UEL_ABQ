@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\tangent_input_20261026\runs\explicit_small"
call "C:\SIMULIA\Commands\abaqus.bat" python "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\scripts\extract_odb.py" > extraction_attempt_01.txt 2>&1
exit /b %errorlevel%

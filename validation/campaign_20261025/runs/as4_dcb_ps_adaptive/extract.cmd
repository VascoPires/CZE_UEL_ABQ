@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\as4_dcb_ps_adaptive"
call "C:\SIMULIA\Commands\abaqus.bat" python "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\scripts\extract_odb.py" > extraction_attempt_01.txt 2>&1
exit /b %errorlevel%

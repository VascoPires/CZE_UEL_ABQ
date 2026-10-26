@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\fixed_dcb_a35p0"
call "C:\SIMULIA\Commands\abaqus.bat" python field_catalog.py > field_catalog.log 2>&1

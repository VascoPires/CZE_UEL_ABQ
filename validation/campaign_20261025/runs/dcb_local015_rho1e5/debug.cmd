@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\runs\dcb_local015_rho1e5"
call "C:\SIMULIA\Commands\abaqus.bat" python debug_kinematics.py > debug_kinematics.txt 2>&1

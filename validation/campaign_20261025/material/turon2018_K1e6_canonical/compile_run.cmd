@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\material\turon2018_K1e6_canonical"
call "C:\Program Files (x86)\Intel\oneAPI\compiler\2023.0.0\env\vars.bat" intel64 vs2022
ifort /nologo /O2 /extend-source:132 driver.f90 kernels.f /exe:material_driver.exe > compiler.txt 2>&1
if errorlevel 1 exit /b 1
material_driver.exe > driver_stdout.txt 2>&1
exit /b %errorlevel%

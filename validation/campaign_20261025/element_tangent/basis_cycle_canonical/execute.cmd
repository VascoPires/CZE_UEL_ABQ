@echo off
cd /d "C:\Users\p2321038\Documents\GitHub\CZE_UEL_ABQ\validation\campaign_20261025\element_tangent\basis_cycle_canonical"
call "C:\Program Files (x86)\Intel\oneAPI\compiler\2023.0.0\env\vars.bat" intel64 vs2022
ifort /nologo /O2 /extend-source:132 driver.f90 test_source.f /exe:element_check.exe > compiler.txt 2>&1
if errorlevel 1 exit /b 1
element_check.exe > execution.txt 2>&1
exit /b %errorlevel%

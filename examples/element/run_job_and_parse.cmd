@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_DIR=%%~fI"
pushd "%REPO_DIR%"
python scripts\run_case.py examples\element\inputs\direction_change.inp --name direction_change_solved --submit
if errorlevel 1 goto failed
python examples\element\parse_elprint_sdv_by_increment.py --datFile runs\direction_change_solved\direction_change_solved.dat --outCsv runs\direction_change_solved\sdv.csv --sdvCount 200
if errorlevel 1 goto failed
python examples\element\plot_sdv_curves.py --csv runs\direction_change_solved\sdv.csv --outPng runs\direction_change_solved\sdv_curves.png
if errorlevel 1 goto failed
popd
exit /b 0
:failed
popd
exit /b 1

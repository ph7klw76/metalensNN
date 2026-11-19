@echo off
setlocal
set "DESIGNER_PYTHONW=%FIBER_METALENS_PYTHONW%"
if not defined DESIGNER_PYTHONW set "DESIGNER_PYTHONW=%USERPROFILE%\anaconda3\pythonw.exe"
if not exist "%DESIGNER_PYTHONW%" (
    echo A Windows Python with Tk and NumPy is needed.
    echo Set FIBER_METALENS_PYTHONW to its full pythonw.exe path and try again.
    pause
    exit /b 1
)
start "Fiber Metalens Designer" "%DESIGNER_PYTHONW%" -B "%~dp0scripts\fiber_metalens_desktop.py"

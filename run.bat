@echo off
REM TTNet 边缘推理 · Windows 一键运行
REM
REM 用法：
REM   run.bat              默认设备（auto）+ 不抽帧
REM   run.bat cpu          强制 CPU
REM   run.bat cpu 4        CPU + 抽帧步长 4（边缘板推荐）
REM
REM 若 python 不在 PATH，可先设置 PYTHON 环境变量指向解释器。

setlocal
cd /d "%~dp0"

set DEVICE=%1
if "%DEVICE%"=="" set DEVICE=auto
set STRIDE=%2
if "%STRIDE%"=="" set STRIDE=1

if "%PYTHON%"=="" set PYTHON=python

echo ==============================================
echo TTNet 裁判 . 边缘推理
echo   设备: %DEVICE%   抽帧步长: %STRIDE%
echo ==============================================
echo.
echo [1/2] 部署自检...
"%PYTHON%" src\selfcheck.py --device %DEVICE%
if errorlevel 1 (
    echo.
    echo 自检未通过，已中止。
    exit /b 1
)

echo.
echo [2/2] 开始推理...
"%PYTHON%" src\infer.py --video videos\test_1.mp4 videos\test_6.mp4 --out results --device %DEVICE% --stride %STRIDE%

echo.
echo 完成。结果在 results\ 目录。
dir /b results
endlocal

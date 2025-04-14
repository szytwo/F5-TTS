@echo off
SET PYTHON_PATH=D:\AI\F5-TTS\venv\Scripts\
REM 设置 GPU 环境变量，选择显卡
set CUDA_VISIBLE_DEVICES=0
REM 执行 Python 脚本
%PYTHON_PATH%python src\f5_tts\infer\infer_fastapi.py
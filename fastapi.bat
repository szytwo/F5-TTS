@echo off
chcp 65001 >nul
echo  启动中，请耐心等待
SET PYTHON_PATH=D:\AI\F5-TTS\venv\Scripts\
REM 设置 GPU 环境变量，选择显卡
SET CUDA_VISIBLE_DEVICES=0
SET ASR_URL=http://127.0.0.1:7868/api/v1/asr
REM 执行 Python 脚本
%PYTHON_PATH%python src\f5_tts\infer\infer_fastapi.py
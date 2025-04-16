import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from func_timeout import func_timeout, FunctionTimedOut
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

from AsrProcessor import AsrProcessor
from AudioProcessor import AudioProcessor
from TextProcessor import TextProcessor
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT
from file_utils import logging, delete_old_files_and_folders

# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有。
# load models
DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# load models
root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
file_dir = str(Path(__file__).resolve().parent)
ckpts_dir = root_dir + "/ckpts"

result_input_dir = root_dir + '/results/input'
result_output_dir = root_dir + '/results/output'
audio_processor = AudioProcessor(result_input_dir, result_output_dir)

vocoder = load_vocoder(is_local=True, local_path=ckpts_dir + "/charactr/vocos-mel-24khz")


def load_f5tts():
    # ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    ckpt_path = ckpts_dir + "/F5TTS_v1_Base/model_1250000.safetensors"
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


F5TTS_ema_model = load_f5tts()


def infer(
        ref_audio_orig,
        ref_text,
        gen_text,
        model,
        remove_silence,
        cross_fade_duration=0.15,
        nfe_step=32,
        speed=1
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)
    final_wave = None
    final_sample_rate = None

    try:
        # 记录开始时间
        start_time = time.time()
        ema_model = F5TTS_ema_model

        # final_wave, final_sample_rate, combined_spectrogram = infer_process(
        #     ref_audio,
        #     ref_text,
        #     gen_text,
        #     ema_model,
        #     vocoder,
        #     cross_fade_duration=cross_fade_duration,
        #     nfe_step=nfe_step,
        #     speed=speed
        # )
        final_wave, final_sample_rate, combined_spectrogram = func_timeout(
            300,  # 超时时间
            infer_process,
            kwargs={
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "gen_text": gen_text,
                "model_obj": ema_model,
                "vocoder": vocoder,
                "cross_fade_duration": cross_fade_duration,
                "nfe_step": nfe_step,
                "speed": speed
            },
        )
        # 计算耗时
        elapsed = time.time() - start_time
        logging.info(f"生成完成，用时: {elapsed}")
    except FunctionTimedOut:
        errcode = -1
        errmsg = "generate_audio 执行超时"
        audio = None
        logging.error(errmsg)
    finally:
        # 删除过期文件
        delete_old_files_and_folders(result_output_dir, 1)
        delete_old_files_and_folders(result_input_dir, 1)
        clear_cuda_cache()

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave), ref_text


def basic_tts(ref_audio_input, ref_text_input, gen_text_input, remove_silence, cross_fade_duration_slider, nfe_slider,
              speed_slider):
    if not ref_text_input:
        asr_processor = AsrProcessor()
        ref_text_input = asr_processor.asr_to_text(ref_audio_input)

    audio_out, ref_text_out = infer(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        tts_model_choice,
        remove_silence,
        cross_fade_duration=cross_fade_duration_slider,
        nfe_step=nfe_slider,
        speed=speed_slider,
    )
    return audio_out, ref_text_out


app = FastAPI(docs_url=None)
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  # 允许跨域的headers，可以用来鉴别来源等作用。
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# 使用本地的 Swagger UI 静态资源
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    logging.info("Custom Swagger UI endpoint hit")
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Custom Swagger UI",
        swagger_js_url="/static/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/5.9.0/swagger-ui.css",
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.get('/test')
async def test():
    """
    测试接口，用于验证服务是否正常运行。
    """
    return PlainTextResponse('success')


@app.post("/zero_shot/")
async def zero_shot(
        prompt_wav: UploadFile = File(
            ...,
            description="选择prompt音频文件，注意采样率不低于16khz"
        ),
        prompt_text: str = Form(
            default="",
            description="请输入prompt文本，需与prompt音频内容一致，空为自动识别"
        ),
        text: str = Form(..., description="输入合成文本"),
        remove_silence: bool = Form(default=False),
        cross_fade_duration: float = Form(default=0.15),
        nfe_steps: int = Form(default=32),
        spaker: float = Form(default=1)
):
    # 保存上传的音频文件
    ref_audio_path = f"results/input/{prompt_wav.filename}"
    with open(ref_audio_path, "wb") as buffer:
        buffer.write(prompt_wav.file.read())

    add_lang_tag = False  # 是否添加语言标签
    text, lang = TextProcessor.ensure_sentence_ends_with_period(text, add_lang_tag)

    if lang == 'zh' or lang == 'zh-cn':  # 中文文本，添加引号，确保不会断句
        keywords = TextProcessor.get_keywords(config_file=file_dir + '/keywords.json')
        text = TextProcessor.replace_chinese_number(text)
        # text = TextProcessor.add_quotation_mark(text, keywords["keywords"], min_length=2)
        text = TextProcessor.replace_pronunciation(text, keywords["cacoepy"])

    print(lang, text)
    # 调用原有的处理函数
    audio_out, ref_text_out = basic_tts(ref_audio_path, prompt_text, text, remove_silence, cross_fade_duration,
                                        nfe_steps, spaker)

    wav_path = audio_processor.generate_wav(audio_out[1], audio_out[0], 0.0, 1.0)
    return JSONResponse({"errcode": 0, "errmsg": "ok", "wav_path": wav_path})


@app.get('/download')
async def download(
        wav_path: str = Query(..., description="输入wav文件路径"),
        name: str = Query(..., description="输入wav文件名")
):
    """
    音频文件下载接口。
    """
    return FileResponse(path=wav_path, filename=name, media_type='application/octet-stream')


# 定义一个函数进行显存清理
def clear_cuda_cache():
    """
    清理PyTorch的显存和系统内存缓存。
    注意上下文，如果在异步执行，会导致清理不了
    """
    logging.info("Clearing GPU memory...")
    # 强制进行垃圾回收
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 重置统计信息
        torch.cuda.reset_peak_memory_stats()
        # 打印显存日志
        logging.info(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9988)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

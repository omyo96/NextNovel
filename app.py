import asyncio
from datetime import datetime
import time

from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File, Form, Response
from typing import List, Dict

from fastapi.responses import StreamingResponse

from gpt import run_openai_chatbot as chatbot
import caption
from diffusion import diffusion_ControlNet
from caption import inference_caption
import torch
import googletrans
import json
import time
import multiprocessing

# import gc
# import GPUtil

app = FastAPI()
pool = multiprocessing.Pool(processes=3)


@app.post('/novel/image')
async def image(image: UploadFile = Form(...)):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes))

    # en_string : 이미지캡셔닝 후 단어로 변환(영어)
    en_string = inference_caption(image_bytes)
    print("11-----------------")
    print(en_string)
    question = f'"{en_string}"\nInterpret this sentence and tell me in one word what object you drew'
    print("22-----------------")
    print(question)

    start = time.time()
    en_string, new_history = chatbot(question, [])
    print("33-----------------")
    print(en_string)

    print(time.time() - start)

    # diffusion 이전 그림 파일 저장
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"diffusion/{current_time}.png"
    img.save(filename)

    start = time.time()
    res = diffusion_ControlNet.creat_image(filename, en_string)
    # res = img
    print(time.time()-start)

    # diffusion 이후 그림 파일 저장
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"diffusion/{current_time}.png"
    # GPUtil.showUtilization()
    res.save(filename)

    # Read the saved image file
    with open(filename, "rb") as f:
        img_bytes = f.read()

    # gc.collect()
    # torch.cuda.empty_cache()

    # Create a streaming response
    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/png"
    )

@app.get('/cuda')
async def hello():
    return torch.cuda.is_available()

@app.get('/')
async def hello2():
    return "hello"

"""
	web.py - Q3 (DCU)
	by Ksuserkqy(20251113620)
	Docs: https://www.ksuser.cn/dcu/
	2025-10-20
"""

import ktse
import uvicorn
from settings import *
from threading import Lock
from fastapi import FastAPI
from contextlib import asynccontextmanager
from idle import IdleManager
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动服务时开启检测
    global g_idle_manager
    g_idle_manager = IdleManager(model_getter=get_model)
    g_idle_manager.start()
    try:
        yield
    finally:
        # 退出时
        if g_idle_manager:
            g_idle_manager.stop()

app = FastAPI(lifespan=lifespan)

g_model: ktse.KsuserTransformersSleepEngine | None = None
g_model_lock = Lock() # 线程锁
g_sleeping = False
g_sleep_level = 1
g_idle_manager: IdleManager | None = None

class CompletionRequest(BaseModel):
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS
    stream: bool = True

def get_model():
    global g_model
    if g_model is None:
        g_model = ktse.KsuserTransformersSleepEngine(model_dir=MODEL_DIR, keep_layers=KEEP_LAYERS)
    return g_model

@app.get("/")
def index():
    return { "message": "KTSE Web Server is running." }

@app.post("/sleep")
def sleep_api():
    global g_sleeping
    with g_model_lock:
        model = get_model()
        model.sleep()
        g_sleeping = True
    return { "code": 200 }

@app.post("/wake_up")
def wake_api():
    global g_sleeping
    with g_model_lock:
        model = get_model()
        model.wake()
        g_sleeping = False
    return { "code": 200 }

@app.get("/is_sleeping")
def is_sleeping_api():
    model = get_model()
    sleeping = False
    try:
        sleeping = model.is_sleeping()
    except Exception:
        sleeping = False
    return { "sleeping": sleeping }

@app.post("/completions")
def completions_api(req: CompletionRequest):
    global g_sleeping
    with g_model_lock:
        model = get_model()
        try:
            model.touch_activity()
        except Exception:
            pass
    if not req.stream:
        with g_model_lock:
            model = get_model()
            model.wake()
            output = model.generate(req.prompt, max_new_tokens=req.max_tokens)
            return {"output": output}
        
    else:
        def stream_gen():
            global g_sleeping
            with g_model_lock:
                model = get_model()
                model.wake()
                for chunk in model.generate_stream(req.prompt, max_new_tokens=req.max_tokens):
                    yield chunk
        return StreamingResponse(stream_gen(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=False)

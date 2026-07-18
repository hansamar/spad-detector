"""FastAPI 应用入口"""

import sys
import os
import logging

# 确保项目根目录在 search path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.api import router


logger = logging.getLogger("spad.backend.validation")


def serializable_validation_errors(exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        cleaned = dict(error)
        ctx = cleaned.get("ctx")
        if isinstance(ctx, dict):
            cleaned["ctx"] = {key: str(value) for key, value in ctx.items()}
        errors.append(cleaned)
    return jsonable_encoder(errors)

app = FastAPI(
    title="SPAD 主动成像仿真 API",
    description="单光子雪崩二极管 (SPAD) 主动照明近距离成像仿真后端服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "file://",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    details = serializable_validation_errors(exc)
    logger.warning("request validation failed: %s %s -> %s", request.method, request.url.path, details)
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

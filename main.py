from fastapi import FastAPI
from app.api.api_v1.routers.llm_router import llm_router
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")


def create_app() -> FastAPI:
    app = FastAPI(title='My FastAPI Application', version='1.0.0')

    # 添加每个 API 路由
    app.include_router(llm_router, prefix='/api/v1/llm')
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True, log_level="debug")

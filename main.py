from fastapi import FastAPI
from app.core.cors import setup_cors


def create_app() -> FastAPI:
    app = FastAPI(title='My FastAPI Application', version='1.0.0')

    # 应用 CORS 设置
    setup_cors(app)

    # 添加 API 路由
    app.include_router(api_router)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

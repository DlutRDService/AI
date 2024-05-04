from fastapi import FastAPI
from app.core.cors import setup_cors
from app.api.api_v1.routers.paper_router import paper_router  # 导入每个模块中的路由器
import uvicorn

def create_app() -> FastAPI:
    app = FastAPI(title='My FastAPI Application', version='1.0.0')

    # 应用 CORS 设置
    setup_cors(app)

    # 添加每个 API 路由
    app.include_router(paper_router, prefix='/api/v1/papers')

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

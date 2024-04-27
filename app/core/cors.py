from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # 或其他适当的前端地址
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
from fastapi.middleware.cors import CORSMiddleware


def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # 前端IP
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
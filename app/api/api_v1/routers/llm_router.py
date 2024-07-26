import httpx
from fastapi import APIRouter, HTTPException, Request
import logging

from app.db.save_to_milvus import embedding
from app.db.milvus_conf import MilvusConf


llm_router = APIRouter()
base_url = "https://ingpt.inspures.com/llm/v1/chat/completions"
logger = logging.getLogger("uvicorn")



@llm_router.post("/chat/")
async def chat_proxy(request: Request):
    client = MilvusConf()
    request_body = await request.json()
    logger.info(f"Received request body: {request_body}")

    vector = embedding(request_body['messages'][1]['content'])
    result = client.sereach([vector], "demo")

    try:
        request_body['messages'][1]['content'] += " 从知识库中查询到如下其他用户的回复提供参考：{}".format(result[0]['human_answers'])
    except HTTPException:
        raise HTTPException(status_code=400, detail="Request body is malformed or missing messages")

    logger.info(f"Received request body: {request_body}")

    headers = {
        "Authorization": "Bearer %HAIYUE_API_KEY%",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(base_url, json=request_body, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="API request failed")

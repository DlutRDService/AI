from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.api.api_v1.pipeline.data_loader import DataLoader

patent_router = APIRouter()


@patent_router.post("/process_plain_text_file/")
async def process_plain_text_file(
    txt: UploadFile = File(...),
    fields: str = Form(''),
    start_index: int = Form(0),
    page_size: int = Form(10)
):
    if fields:
        fields = fields.split(', ')

    result = {}

    # Make sure the type of file is txt
    if not txt.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .txt files are accepted.")

    # 异步读取
    papers = await txt.read()
    # 将内容转换为字符串
    papers = papers.decode("utf-8")

    data_loader = DataLoader(papers, fields)
    for i in range(start_index, start_index + page_size):
        result[i] = data_loader[i].to_dict()

    return result

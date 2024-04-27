from fastapi import APIRouter, HTTPException, UploadFile, File
from ..services.paper_service import

router = APIRouter()

@router.post("/process_plain_text_file/")
async def process_plain_text_file(file: UploadFile = File(...)):
    # Make sure the type of file is txt
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .txt files are accepted.")

    content = await file.read()
    # 处理文件内容
    processed_content = (content)  # 假设你有一个函数来处理文件内容

    return {"filename": file.filename, "content": processed_content}

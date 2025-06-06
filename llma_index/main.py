from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import ChatPromptTemplate, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from uuid import uuid4
from pathlib import Path
import os
from pydantic import BaseModel

from helper import (
    get_system_prompt,
    load_llm,
    get_prompts,
    get_chat_text_qa_msgs,
    get_chat_refine_msgs,
    manage_index,
    run_query
)

app = FastAPI()

UPLOAD_DIR = "uploaded_docs"
SAVE_DIR = "folder1.1"
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(SAVE_DIR).mkdir(exist_ok=True)

# In-memory cache
document_cache = {}

# Load model and config once
system_prompt = get_system_prompt()
llm = load_llm()
qa_prompt_str, refine_prompt_str = get_prompts()

chat_text_qa_msgs = get_chat_text_qa_msgs()
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

chat_refine_msgs = get_chat_refine_msgs()
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

embed_model = "local:BAAI/bge-small-en-v1.5"
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    raw_docs = SimpleDirectoryReader(input_dir=UPLOAD_DIR, num_files_limit=100).load_data()
    combined_doc = Document(text="\n\n".join([doc.text for doc in raw_docs]))

    document_cache["latest"] = combined_doc

    manage_index(combined_doc, embed_model, node_parser, save_dir=SAVE_DIR)

    return JSONResponse(content={"message": "File uploaded and indexed", "chars": len(combined_doc.text)})



class QueryRequest(BaseModel):
    query: str

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    if "latest" not in document_cache:
        return JSONResponse(status_code=400, content={"error": "No document uploaded yet."})

    documents = document_cache["latest"]

    result = run_query(
        documents,
        embed_model,
        node_parser,
        SAVE_DIR,
        text_qa_template,
        refine_template,
        request.query
    )

    return JSONResponse(content={"question": request.query, "answer": str(result)})


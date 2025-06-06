from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login


document=[]
for file in os.listdir("docs"):
  if file.endswith(".pdf"):
    pdf_path="./docs/"+file
    loader=PyPDFLoader(pdf_path)
    document.extend(loader.load())
  elif file.endswith('.docx') or file.endswith('.doc'):
    doc_path="./docs/"+file
    loader=Docx2txtLoader(doc_path)
    document.extend(loader.load())
  elif file.endswith('.txt'):
    text_path="./docs/"+file
    loader=TextLoader(text_path)
    document.extend(loader.load())

document_splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)


document_chunks=document_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



# opeaikey

vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./data')
vectordb.persist()



# Load env vars
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("❌ No Hugging Face token found")



# Login to Hugging Face
login(hf_token)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-fp16",
                                          use_auth_token=True,)


model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-fp16",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                              #load_in_8bit=True,
                                              load_in_4bit=True
                                             )

pipe=pipeline("text-generation",
              model=model,
              tokenizer=tokenizer,
              torch_dtype=torch.bfloat16,
              device_map='auto',
              max_new_tokens=512,
              min_new_tokens=-1,
              top_k=30

              )

llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})
memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)

#Create our Q/A Chain
pdf_qa=ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=vectordb.as_retriever(search_kwargs={'k':6}),
                                             verbose=False, memory=memory)

result=pdf_qa({"question":"who is shivaji maharaj?"})

print(result['answer'])
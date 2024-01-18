from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ftlangdetect import detect
from serverinfo import si
import torch
from transformers import AutoTokenizer
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer,BartForConditionalGeneration
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os

from langchain_community.document_loaders import PyPDFLoader
from duckduckgo_search import DDGS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import WikipediaLoader

search = DDGS()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)

embeddings = HuggingFaceEmbeddings(
    model_name="distiluse-base-multilingual-cased-v1", # "BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

class Param(BaseModel):
  prompt : str
  type = "PL"

class Chat(BaseModel):
  prompt : str
  history : list
  lang = "auto"
  type = "AI assist"
  temp = 0.5
  top_p = 1.0
  top_k = 0
  max = 1024

to = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

origins = [
    "http://canvers.net",
    "https://canvers.net",   
    "http://www.canvers.net",
    "https://www.canvers.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/")
def main():
  return { "result" : True, "data" : "CIRCULUS-VECTOR V1" }      

@app.get("/monitor")
def monitor():
  return si.getAll()

@app.get("/v1/language", summary="어느 언어인지 분석합니다.")
def language(input : str):
  return { "result" : True, "data" : detect(input)['lang'] }

@app.get("/v1/search", summary="url로 부터 입력")
def search(prompt="", userId="test", projectId="test"): #max=20480): # gen or med
  vecs = Chroma(persist_directory=f"./{userId}_{projectId}", embedding_function=embeddings)
  docs = vecs.similarity_search_with_relevance_scores(prompt, k=3, score_threshold=0.5)

  return { "result" : True, "data" : docs }

@app.post("/v1/fromFile", summary="파일로 부터 입력")
def fromFile(file : UploadFile = File(...), userId="test", projectId="test"):

  type = os.path.splitext(file.filename)[1].replace(".","").lower()

  temp_file = ""

  if type == 'pdf':
    loader = PyPDFLoader(temp_file)
  elif type == "doc" or type == "docx": 
    loader = Docx2txtLoader(temp_file)
  elif type == "htm" or type == "html": 
    loader = BSHTMLLoader(temp_file)
  elif type == "ppt" or type == "pptx": 
    loader = UnstructuredPowerPointLoader(temp_file)  
  elif type == "hwp" or type == "hwpx": 
    test = ""
  elif type == "xls" or type == "xlsx":
    loader = UnstructuredExcelLoader(temp_file, mode="elements")
  elif type == "csv":
    loader = CSVLoader(file_path=temp_file)
  else: # txt 및 기타 파일로 간주
    loader = UnstructuredFileLoader(temp_file)

  documents = loader.load_and_split()
  chunks = text_splitter.split_documents(documents)

  vecs = Chroma.from_documents(chunks, embeddings, persist_directory=f"./{userId}_{projectId}")

  return { "result" : True, "data" : len(documents)}

@app.get("/v1/fromUrl", summary="url로 부터 입력")
def fromUrl(url="", userId="test", projectId="test"): #max=20480): # gen or med

  temp_file = ""

  if url in "youtube":
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
  elif url in "wikipedia":
    loader = WikipediaLoader(query="HUNTER X HUNTER", load_max_docs=2)
  else: # txt 파일로 간주
    loader = WebBaseLoader(url)

  documents = loader.load_and_split()
  chunks = text_splitter.split_documents(documents)
  vecs = Chroma.from_documents(chunks, embeddings, persist_directory=f"./{userId}_{projectId}")

  return { "result" : True, "data" : len(documents)}

  return { "result" : True }


print("Loading Complete!")
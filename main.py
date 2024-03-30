from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
#from ftlangdetect import detect
from serverinfo import si
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer,BartForConditionalGeneration
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma
import zipfile
import requests
import json 
from urllib import parse

text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1024, chunk_overlap=128)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", #"BAAI/bge-large-en-v1.5", #"BAAI/bge-m3", # # distiluse-base-multilingual-cased-v1
    model_kwargs={"device": "cuda"}, # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    encode_kwargs={"normalize_embeddings": True},
)

class Param(BaseModel):
  prompt : str
  type : str

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
  return { "result" : True, "data" : input }

@app.get("/v1/find", summary="입력된 데이터 조회")
def search(prompt="", userId="test", projectId="test"): #max=20480): # gen or med
  print(prompt)
  vecs = Chroma(persist_directory=f"./db/{userId}_{projectId}", embedding_function=embeddings)
  docs = vecs.similarity_search_with_relevance_scores(prompt, k=3) # , score_threshold=0.5
  print(docs)
  return { "result" : True, "data" : docs }

@app.get("/v1/query", summary="언어모델과 연동 테스트")
def search(prompt="", userId="test", projectId="test", type="매우 친절한 인공지능으로 모르면 모른다고 하고 아는건 최대한 자세히 말해주세요."): #max=20480): # gen or med
  vecs = Chroma(persist_directory=f"./db/{userId}_{projectId}", embedding_function=embeddings)
  docs = vecs.similarity_search_with_relevance_scores(prompt, k=2) # , score_threshold=0.5
  data = ''

  for doc in docs:
    data = data + doc[0].page_content + "\n"

  data = data.replace('\n', ' ').replace('\r', '')

  print(data)

  res = requests.post('https://oe-napi.circul.us/v1/rag2chat', json={ "prompt" : prompt, "type" : type, "history" : [], "rag" : data })

  print(res.text)
 
  return { "result" : True, "data" : res.text }

@app.post("/v1/fromFile", summary="파일로 부터 입력")
def fromFile(file : UploadFile = File(...), userId="test", projectId="test"):

  type = os.path.splitext(file.filename)[1].replace(".","").lower()

  fo = open('./uploads/' + file.filename, "wb+")
  fo.write(file.file.read())

  temp_file = './uploads/' + file.filename


  print("preparing", temp_file)

  if type == 'pdf':
    loader = PyPDFLoader(temp_file)
  elif type == "doc" or type == "docx":
    type = "doc" 
    loader = Docx2txtLoader(temp_file)
  elif type == "htm" or type == "html": 
    type = "html"
    loader = BSHTMLLoader(temp_file)
  elif type == "ppt" or type == "pptx":
    type = "ppt" 
    loader = UnstructuredPowerPointLoader(temp_file)  
  elif type == "hwp": # or type == "hwpx
    print(f"hwp5html --output ./hwp {temp_file}")
    os.system(f'hwp5html --html --output ./hwp/index.html "{temp_file}"')
    loader = BSHTMLLoader("./hwp/index.html",open_encoding='UTF8')
  elif type == "hwpx": # 압축 파일
    type = "hwp"
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
      zip_ref.extractall("./zip")
    loader = UnstructuredXMLLoader('./zip/Contents/section0.xml')
  elif type == "xls" or type == "xlsx":
    type = "xls"
    loader = UnstructuredExcelLoader(temp_file, mode="elements")
  elif type == "csv":
    loader = CSVLoader(file_path=temp_file)
  else: # txt 및 기타 파일로 간주
    type = "txt"
    loader = UnstructuredFileLoader(temp_file)

  print("loading", temp_file)

  documents = loader.load_and_split()
  print("splitting", temp_file)
  chunks = text_splitter.split_documents(documents)

  print("saving...", temp_file)


  vecs = Chroma.from_documents(chunks, embeddings, persist_directory=f"./db/{userId}_{projectId}")

  return { "result" : True, "data" : len(documents)}

@app.get("/v1/fromUrl", summary="url로 부터 입력")
def fromUrl(url="", userId="test", projectId="test", lang='ko'): #max=20480): # gen or med
  if os.path.exists(f"./db/{userId}_{projectId}"):
    os.remove(f"./db/{userId}_{projectId}")
  temp_file = ""
  type = 'web'

  print("preparing...", url)

  if "youtube" in url:
    type = "youtube"
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True) # , language=[lang, "id"], translation=lang
    print(loader)
  elif "wikipedia" in url :
    type = "wikipedia"
    query = url.split("/wiki/")[1]
    print("wiki test",query)
    if "ko.wikipedia" in url:
      query = parse.unquote(query, encoding="utf-8")
      print("korean search", query)
      loader = WikipediaLoader(query=query, lang="ko", load_max_docs=2)
    else:
      loader = WikipediaLoader(query=query, lang="en", load_max_docs=2)
  else: # 일반 웹으로 간주
    print('general process')
    loader = WebBaseLoader(url)

  print("loading...", projectId)
  documents = loader.load_and_split()

  print("splitting...",documents)
  chunks = text_splitter.split_documents(documents)

  print("saving...",chunks)

  vecs = Chroma.from_documents(chunks, embeddings, persist_directory=f"./db/{userId}_{projectId}")

  print(documents)
  return { "result" : True, "data" : len(documents)}

print("Loading Complete!")
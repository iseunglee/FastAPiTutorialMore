from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io

app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        # 이미지 파일 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 이미지를 그레이스케일로 변환
        gray_image = image.convert('L')

        # 변환된 이미지를 byte로 변환
        img_byte_arr = io.BytesIO()
        gray_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # StreamingResponse로 이미지 반환
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")

@app.get("/")
def read_root():
    return {"Hello": "Lion"}

#################### 이미지 처리 연습문제제
# @app.post("/rotate/")
# async def rotate_image(file : UploadFile = File()):
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail='Invalid')

#     image_data = await file.read()
#     image = Image.open(io.BytesIO(image_data))

#     # 이미지를 회전
#     rotated_image = image.rotate(90, expand=True)

#     # 변환된 이미지를 byte로 변환
#     img_byte_arr = io.BytesIO()
#     rotated_image.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()

#     return StreamingResponse(io.BytesIO(img_byte_arr), media_type='image/png')
@app.post("/rotate/")
async def rotate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 이미지를 90도 회전
    rotated_image = image.rotate(90, expand=True)

    img_byte_arr = io.BytesIO()
    rotated_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")

################# 머신러닝

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# 모델 로드
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

class IrisModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict_iris/")
def predict_iris(iris: IrisModel):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

################### 와인 데이터셋
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

class WineModel(BaseModel):
    features : list

# 내코드
@app.post("/predict_wine/")
def predict_wine(wine : WineModel):
    # 예외처리 한 코드 
    try:
        data = np.array([wine.features])
        prediction = model.predict(data)
        return {"prediction" : int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 강사님 코드
# @app.post("/predict/")
# def predict_wine_quality(wine: WineFeatures):
#     try:
#         prediction = model.predict([wine.features])
#         return {"prediction": int(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

################# 언어모델 api
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english") # 언어 -> 토큰 
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english") # 모델 가져옴

class TextData(BaseModel): # 텍스트 데이터 모델 생성
    text: str 

@app.post("/classify/")
async def classify_text(data: TextData):
    inputs = tokenizer(data.text, return_tensors="pt") # 입력받은 텍스트 데이터를 토큰화
    with torch.no_grad():
        logits = model(**inputs).logits # 이진분류문제에 쓰이는 logit (logistic + probit)

        predicted_class_id = logits.argmax().item()
        model.config.id2label[predicted_class_id]
    return {"result": predicted_class_id}

#################### 벡터 DB 이용한 자연어 검색
import pandas as pd
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

# 데이터 로드
books = pd.read_excel('science_books.xlsx')

# 임베딩 모델 초기화
sbert = SentenceTransformerEmbeddings(model_name='jhgan/ko-sroberta-multitask')

# 벡터 저장소 생성
vector_store = Chroma.from_texts(
    texts=books['제목'].tolist(),
    embedding=sbert
)

@app.post("/search/")
def search_books(query: str):
    results = vector_store.similarity_search(query=query, k=3)  # 상위 3개 결과 반환
    return {"query": query, "results": results}
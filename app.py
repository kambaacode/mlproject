from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request

import numpy as np 
import pandas as pd 

from src.pipeline.predict_pipeline import PredictPipeline, StudentInput

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")




@app.post("/predict")
def predict_endpoint(data: StudentInput):
    df = pd.DataFrame([data.dict()])

    predictPipeline = PredictPipeline()
    result = predictPipeline.predict(df)

    return {"prediction": result[0]}
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from helpers import predict_image, read_imagefile

app = FastAPI(title='SolarClassifier')

@app.get("/")
def read_root():
  return {"Hello":"World"}

@app.post("/predict", status_code=200)
async def predict_img(file: UploadFile = File(...)):
  extension = file.filename.split(".")[-1] in ("jpg","jpeg","png")
  if not extension:
      return "Image must be jpg or png format!"
  image = read_imagefile(await file.read())
  prediction = predict_image(image)
    
  return {"prediction" : prediction}

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
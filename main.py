from fastapi import FastAPI
from pydantic import BaseModel
from ml.img_handler import imageDecoder, deleteImage
from ml.predict import predict_image

class Item(BaseModel):
    img: str

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "functioning"}

@app.post("/pred")
async def root(item: Item):
    imageDecoder(item.img)
    pred = predict_image()
    print(pred.item())
    deleteImage()
    return {"prediction": "Spoof" if pred == 1 else "Live"}
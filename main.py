import base64
import numpy as np
from PIL import Image
from io import BytesIO
from src.predict import predict
from fastapi import FastAPI
from src.model import load_model
from pydantic import BaseModel

class Item(BaseModel):
    img: str

app = FastAPI()
model = load_model()
model.eval()

@app.get("/")
async def root():
    return {"status": "functioning"}

@app.post("/pred")
async def root(item: Item):
    img_b = base64.b64decode(item.img)
    img = Image.open(BytesIO(img_b))
    pred = predict(model, np.array(img))
    return {"prediction": "Spoof" if pred == 1 else "Live"}
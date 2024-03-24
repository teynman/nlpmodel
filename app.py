from fastapi import FastAPI
from fastai.text.all import load_learner
from pydantic import BaseModel

app = FastAPI()

# Load the model
model = load_learner('model.pkl')

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    prediction = model.predict(review.text)
    # Assuming the prediction is a tuple and the first element is the category
    result = prediction[0]
    return {"sentiment": result}
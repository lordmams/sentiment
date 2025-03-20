from fastapi import FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from pydantic import BaseModel
from sentiment import Main

# Create FastAPI app
app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

# Define input model
class SentimentRequest(BaseModel):
    text: str

# Create global instance of Main class
sentiment_analyzer = Main()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    try:
        sentiment = sentiment_analyzer.predict(request.text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
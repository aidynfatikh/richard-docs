from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

#fake ml lol
@app.post("/predict")
async def predict(request: Request):
    return {
        "boxes": [
            {
                "x1": 50,
                "y1": 40,
                "x2": 200,
                "y2": 300,
                "confidence": 0.75,
                "class": "object"
            }
        ],
        "model_version": "mock_v1",
        "latency_ms": 5
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
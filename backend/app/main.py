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
        "document_id": "doc_mock_001",
        "pages": [
            {
                "page_number": 1,
                "size": {
                    "width_px": 2480,
                    "height_px": 3508
                },
                "stamps": [
                    {
                        "id": "stamp_1",
                        "bbox": [320, 2200, 900, 2800],
                        "confidence": 0.94,
                        "type": "round_blue",
                        "text": "ТОО «Компания»"
                    }
                ],
                "signatures": [
                    {
                        "id": "sign_1",
                        "bbox": [1400, 2300, 2100, 2700],
                        "confidence": 0.91,
                        "role": "director",
                        "is_digital": False
                    }
                ],
                "qrs": [
                    {
                        "id": "qr_1",
                        "bbox": [1800, 200, 2300, 700],
                        "confidence": 0.97,
                        "decoded_data": "https://example.com/verify?id=12345",
                        "version": "QR-Model-v1"
                    }
                ]
            }
        ],
        "summary": {
            "total_pages": 1,
            "total_stamps": 1,
            "total_signatures": 1,
            "total_qrs": 1
        },
        "meta": {
            "model_version": "doc-detector-v0.1-mock",
            "inference_time_ms": 23
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
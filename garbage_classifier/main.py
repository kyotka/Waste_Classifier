import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from model import GarbageClassifier, CLASS_INFO

app = FastAPI(
    title="Waste Classification API",
    description="Computer Vision API for automated waste sorting using ResNet-50",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_resnet50_garbage.pth")
classifier = None

@app.on_event("startup")
def load_classifier():
    global classifier
    classifier = GarbageClassifier(checkpoint_path=MODEL_PATH)

@app.get("/api/info")
def get_model_info():
    return {
        "architecture": "ResNet-50",
        "num_classes": len(classifier.class_names) if classifier else 6,
        "classes": classifier.class_names if classifier else [],
        "categories": CLASS_INFO,
        "resolution": "224x224 RGB",
        "accuracy": f"{classifier.best_acc:.2f}%" if classifier and classifier.best_acc else "96.84%",
        "framework": "PyTorch",
        "device": str(classifier.device) if classifier else "cpu"
    }

@app.post("/api/predict")
async def predict_garbage(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected an image."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = classifier.predict(image_bytes)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    return HTMLResponse("<h1>API is running.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

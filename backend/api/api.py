from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from services.pipeline.run_pipeline import run_full_pipeline, extract_image_number

app = FastAPI()

# Enable CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/run-pipeline/")
async def run_pipeline(
    file: UploadFile,
    threshold: int = Form(2000),
):
    # Save uploaded file to disk
    upload_dir = "./data/module-1/images"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    image_no = extract_image_number(file_path)

    # Run the pipeline
    result = run_full_pipeline(image_path=file_path, image_no=image_no, threshold=threshold)

    # Collect character image paths
    output_dir = f"./results/module1/image_{image_no}"
    char_images = [
        f"/images/module1/image_{image_no}/{f}"
        for f in os.listdir(output_dir)
        if f.endswith(".png") and "_character_" in f
    ]
    
    return JSONResponse(content={"characters": char_images})

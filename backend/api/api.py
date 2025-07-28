from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import shutil
from services.pipeline.run_pipeline import run_full_pipeline, extract_image_number
from services.module2.service import run_module2

app = APIRouter()  # ⬅️ Use APIRouter instead of FastAPI()

@app.post("/api/run-pipeline/")
async def run_pipeline(
    file: UploadFile,
    threshold: int = Form(2000),
):
    upload_dir = "./data/module-1/images"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image_no = extract_image_number(file_path)

    result = run_full_pipeline(image_path=file_path, image_no=image_no, threshold=threshold)

    output_dir = f"./results/module1/image_{image_no}"
    char_images = [
        f"/images/module1/image_{image_no}/{f}"
        for f in os.listdir(output_dir)
        if f.endswith(".png") and "_character_" in f
    ]

    return JSONResponse(content={
        "image_no": image_no,
        "characters": char_images,
        "denoised_image": f"/images/module1/image_{image_no}/denoised_image.png",
        "final_image": f"/images/module1/image_{image_no}/final_image_second_pass.png"
    })

@app.post("/api/run-module2/")
async def run_module2_endpoint(
    image_no: int = Form(...),
    segmented_images: list = Form(...),
):
    total_chars = len(segmented_images)

    base_dir = "./results/module2"
    module2_dir = os.path.join(base_dir, f"image_{image_no}")
    os.makedirs(module2_dir, exist_ok=True)

    for i, img_path in enumerate(segmented_images):
        src_path = img_path.replace("/images", ".")
        dst_filename = f"{image_no}_image_character_{i+1}.png"
        dst_path = os.path.join(module2_dir, dst_filename)
        shutil.copy(src_path, dst_path)

    predictions = run_module2(image_no=image_no, total_chars=total_chars, user_need="both")

    return JSONResponse(content={
        "image_no": image_no,
        "predictions": predictions
    })

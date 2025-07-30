from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import shutil

from services.pipeline.run_pipeline import run_full_pipeline, extract_image_number
from services.module2.service import run_module2
from services.module3.service import run_module3

app = APIRouter()  # ⬅️ Use APIRouter instead of FastAPI()

@app.post("/api/run-pipeline/")
async def run_pipeline(
    file: UploadFile,
    threshold: int = Form(2000),
):
    # Define directories
    upload_dir = "./data/module-1/images"
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract image number
    image_no = extract_image_number(file_path)
    output_dir = f"../results/module-1/image_{image_no}"
    print(f"Output directory: {output_dir}")
    # Run the pipeline
    result = run_full_pipeline(image_path=file_path, image_no=image_no, threshold=threshold)

    char_images = [
        f"../results/module-1/image_{image_no}/{f}"
        for f in os.listdir(output_dir)
        if f.endswith(".png") and "_character_" in f
    ]
    char_images = [
        path.replace("../results", "/images") for path in char_images
    ]
    return JSONResponse(content={
        "message": "Pipeline completed successfully",
        "image_no": image_no,
        "char_images": char_images,
        "raw_image": f"/images/module-1/image_{image_no}/{file.filename}",
    })


    # # Source directory for images (where pipeline saves them)
    # output_dir = f"../results/module-1/image_{image_no}"

    # # Target directory in root (relative to backend/api/api.py)
    # target_dir = "../results/module-1"  # Adjusted path to root/results/module1
    # os.makedirs(target_dir, exist_ok=True)

    # # List of character images
    # char_images = [
    #     f"/images/module-1/image_{image_no}/{f}"
    #     for f in os.listdir(output_dir)
    #     if f.endswith(".png") and "_character_" in f
    # ]

    # # Files to copy
    # files_to_copy = [
    #     os.path.join(output_dir, f) for f in os.listdir(output_dir)
    #     if f.endswith(".png") and ("_character_" in f or f in ["denoised_image.png", "final_image_second_pass.png"])
    # ]

    # # Copy files to target directory
    # for file_path in files_to_copy:
    #     file_name = os.path.basename(file_path)
    #     target_path = os.path.join(target_dir, file_name)
    #     shutil.copy(file_path, target_path)

    # # Return response with paths relative to the original output directory
    # return JSONResponse(content={
    #     "image_no": image_no,
    #     "characters": char_images,
    #     "denoised_image": f"/images/module-1/image_{image_no}/denoised_image.png",
    #     "final_image": f"/images/module-1/image_{image_no}/final_image_second_pass.png"
    # })

@app.post("/api/run-module2/")
async def run_module2_endpoint(
    image_no: int = Form(...),
    total_chars: int = Form(...),
):
    print(image_no, total_chars, "image_no and total_chars in run_module2_endpoint")
    base_dir = "./results/module2"
    module2_dir = os.path.join(base_dir, f"image_{image_no}")
    os.makedirs(module2_dir, exist_ok=True)

    # for i, img_path in enumerate(segmented_images):
    #     src_path = img_path.replace("/images", ".")
    #     dst_filename = f"{image_no}_image_character_{i+1}.png"
    #     dst_path = os.path.join(module2_dir, dst_filename)
    #     shutil.copy(src_path, dst_path)
 
    predictions = run_module2(image_no=image_no, total_chars=total_chars, user_need="both")

    return JSONResponse(content={
        "image_no": image_no,
        "predictions": predictions
    })


@app.post("/api/run-module3/")
async def run_module2_endpoint(
    final_sequence: str = Form(...),
):
    print(final_sequence, "final sequence to module3")
    # base_dir = "./results/module2"
    # module2_dir = os.path.join(base_dir, f"image_{image_no}")
    # os.makedirs(module2_dir, exist_ok=True)

    # for i, img_path in enumerate(segmented_images):
    #     src_path = img_path.replace("/images", ".")
    #     dst_filename = f"{image_no}_image_character_{i+1}.png"
    #     dst_path = os.path.join(module2_dir, dst_filename)
    #     shutil.copy(src_path, dst_path)
 
    predictions = run_module3(final_sequence=final_sequence)
    return JSONResponse(content={
        "final_sequence": final_sequence,
        "predictions": predictions,
        "best_words": predictions["best"]["words"]
    })


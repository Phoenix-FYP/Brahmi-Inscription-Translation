import os
from .main import process_image

def run_module1(image_path: str, image_no: int = 1, threshold: int = 2000, output_base: str = "../frontend/public/images/module-1/",output_base2:str="../results/module-1") -> dict:
    """
    Handles output path setup and delegates to the module-1 processing logic.
    """
    # Ensure output path
    output_dir = os.path.join(output_base, f"image_{image_no}")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    output_dir2 = os.path.join(output_base2, f"image_{image_no}")
    print(f"Output directory for second pass: {output_dir2}")
    os.makedirs(output_dir2, exist_ok=True)

    # Run the image processing pipeline
    result = process_image(image_path=image_path, output_dir=output_dir, image_no=image_no, threshold=threshold,output_base2=output_base2)

    # Add output folder path to result
    result.update({
    "output_dir": output_dir,
    "output_dir2": output_dir2
     })
    print(f"Module 1 processing complete. Output saved to {output_dir} and {output_dir2}")
    return result

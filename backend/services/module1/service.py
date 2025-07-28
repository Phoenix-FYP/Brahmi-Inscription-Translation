import os
from .main import process_image

def run_module1(image_path: str, image_no: int = 1, threshold: int = 2000, output_base: str = "../frontend/public/images/module-1/") -> dict:
    """
    Handles output path setup and delegates to the module-1 processing logic.
    """
    # Ensure output path
    output_dir = os.path.join(output_base, f"image_{image_no}")
    os.makedirs(output_dir, exist_ok=True)

    # Run the image processing pipeline
    result = process_image(image_path=image_path, output_dir=output_dir, image_no=image_no, threshold=threshold)

    # Add output folder path to result
    result["output_dir"] = output_dir
    return result

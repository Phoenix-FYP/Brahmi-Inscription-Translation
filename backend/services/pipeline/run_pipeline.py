import os
import shutil
from services.module1.service import run_module1
from services.module2.service import run_module2

def run_full_pipeline(image_path: str, image_no: int, threshold: int = 2000):
    print("🔧 [Pipeline] Starting Module 1: Preprocessing and Segmentation...")

    # Run Module 1
    result = run_module1(image_path=image_path, image_no=image_no, threshold=threshold)
    output_dir = result["output_dir"]
    output_dir2 = result["output_dir2"]

    # Get list of saved character images
    char_image_paths = [
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.endswith(".png") and "_character_" in f
    ]
    char_image_paths.sort() 

    char_image_paths2 = [
        os.path.join(output_dir2, f) for f in os.listdir(output_dir2)
        if f.endswith(".png") and "_character_" in f
    ]
    char_image_paths2.sort() 
   
    if not char_image_paths:
        print("No character segments returned from Module 1")
        return

    total_chars = len(char_image_paths)

    # module3_dir = os.path.join("../results/module-1", f"image_{image_no}")
    # os.makedirs(module3_dir, exist_ok=True)
    # print(char_image_paths, "char_image_paths2 in here")
    # for i, char_path in enumerate(char_image_paths2):
    #     dst_filename = f"{image_no}_image_character_{i+1}.png"
    #     dst_path = os.path.join(module3_dir, dst_filename)
    #     shutil.copy(char_path, dst_path)

    print(f"Module 1 complete. {total_chars} character(s) segmented.")
    print("➡ Copying characters to Module 2 directory...")

    # Set Module 2 output directory
    module2_dir = os.path.join("./results/module2", f"image_{image_no}")
    os.makedirs(module2_dir, exist_ok=True)

    for i, char_path in enumerate(char_image_paths2):
        dst_filename = f"{image_no}_image_character_{i+1}.png"
        dst_path = os.path.join(module2_dir, dst_filename)
        shutil.copy(char_path, dst_path)

   

    print("Characters copied. Moving to Module 2...")
    return
    # Run Module 2
    # predictions = run_module2(image_no=image_no, total_chars=total_chars, user_need="both")

    # print("Module 2 complete.")
    # return predictions

def extract_image_number(image_path: str) -> int:
    """
    Extracts the image number from a filename assuming format like '12.jpg' or 'image_12.jpg'.
    Returns integer image number or raises ValueError.
    """
    base = os.path.basename(image_path)  # e.g., "12.jpg"
    name, ext = os.path.splitext(base)  # name="12", ext=".jpg"
    
    # If filename is purely a number like "12"
    if name.isdigit():
        return int(name)
    
    # If filename contains number like "image_12"
    import re
    match = re.search(r'(\d+)', name)
    if match:
        return int(match.group(1))
    
    raise ValueError(f"Could not extract image number from filename: {base}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.abspath(os.path.join(BASE_DIR, '../../../data/module-1/images/12.jpg'))
    image_no = extract_image_number(image_path)
    threshold = 2000  # <-- Now passed to Module 1

    final_output = run_full_pipeline(image_path=image_path, image_no=image_no, threshold=threshold)

    print("\n Final Output from Pipeline:")
    if isinstance(final_output, dict):
        final_sequence = final_output.get("final_sequence", "")
        print(f"Predicted Sinhala character sequence: {final_sequence}")
    else:
        print("⚠ Unexpected result format.")

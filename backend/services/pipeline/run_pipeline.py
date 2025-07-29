import os
import shutil
from services.module1.service import run_module1
from services.module2.service import run_module2
from services.module3.service import run_module3
from services.module4.service import run_module4
import ast  # To safely parse the string list from result.txt


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

    # print("Module 2 complete."

    # # # Save Module 2 result to predictions.txt
    # module3_dir = os.path.join("./results/module3")
    # os.makedirs(module3_dir, exist_ok=True)

    # sequence_str = predictions.get("final_sequence", "")
    # pred_file_path = os.path.join(module3_dir, "predictions.txt")
    # with open(pred_file_path, "w", encoding="utf-8") as f:
    #     f.write(sequence_str.strip())

    # print("➡ Running Module 3: Grapheme Segmentation and Correction...")

    # # Read and process with Module 3
    # with open(pred_file_path, "r", encoding="utf-8") as f:
    #     raw_text = f.read().strip()

    # if not raw_text:
    #     print("⚠ No text found in predictions.txt for Module 3")
    #     return predictions

    # module3_result = run_module3(raw_text)

    # print("Module 3 complete.")
    # print("\nFinal Segmented Output:")
    # print("Segmented:", " ".join(module3_result["best"]["words"]))
    # if module3_result["needs_correction"]:
    #     print("Corrected:", module3_result["corrected"])

    # # Save Module 3 result to result.txt (needed for Module 4)
    # result_path = os.path.join(module3_dir, "result.txt")
    # with open(result_path, "w", encoding="utf-8") as f:
    #     f.write(str(module3_result["best"]["words"]))  # Write as Python list string

    # print("➡ Running Module 4: Grammatical Reordering and Correction...")

    # # Read and parse result.txt to get Brahmi word list
    # with open(result_path, "r", encoding="utf-8") as f:
    #     words_str = f.read().strip()
    #     try:
    #         brahmi_words = ast.literal_eval(words_str)
    #     except Exception as e:
    #         print(f"⚠ Failed to parse result.txt: {e}")
    #         brahmi_words = []

    # if not isinstance(brahmi_words, list) or not brahmi_words:
    #     print("⚠ No valid Brahmi word list found for Module 4")
    #     return {
    #         "module2_prediction": predictions,
    #         "module3_segmentation": module3_result,
    #     }

    # module4_result = run_module4(brahmi_words)

    # print("Module 4 complete.")
    # print("\nFinal Reordered Sentence:")
    # print("Reordered:", module4_result.get("corrected_sentence", "N/A"))

    # # Combine all results
    # return {
    #     "module2_prediction": predictions,
    #     "module3_segmentation": module3_result,
    #     "module4_output": module4_result
    # }


def extract_image_number(image_path: str) -> int:
    """
    Extracts the image number from a filename assuming format like '12.jpg' or 'image_12.jpg'.
    Returns integer image number or raises ValueError.
    """
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)

    if name.isdigit():
        return int(name)

    import re
    match = re.search(r'(\d+)', name)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not extract image number from filename: {base}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.abspath(os.path.join(BASE_DIR, '../../../data/module-1/images/61.jpg'))
    image_no = extract_image_number(image_path)
    threshold = 1000

    final_output = run_full_pipeline(image_path=image_path, image_no=image_no, threshold=threshold)

    print("\n✅ Final Output from Pipeline:")
    if isinstance(final_output, dict):
        pred_seq = final_output["module2_prediction"].get("final_sequence", "")
        print(f"Module 2 Prediction: {pred_seq}")
        print("Module 3 Segmentation:", " ".join(final_output["module3_segmentation"]["best"]["words"]))
        if final_output["module3_segmentation"]["needs_correction"]:
            print("Module 3 Corrected:", final_output["module3_segmentation"]["corrected"])
        print("Module 4 Reordered:", final_output["module4_output"].get("corrected_sentence", "N/A"))
    else:
        print("⚠ Unexpected result format.")

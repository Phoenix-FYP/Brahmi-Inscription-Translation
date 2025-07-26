# backend/services/module2/main.py

import os
from .main_predictor import MainPredictor

def process_characters(image_no: int, total_chars: int, user_need: str = "both", base_path: str = "./results/module-2") -> dict:
    """
    Processes character images for a given image number and character count.
    Returns final sequence and detailed prediction results.
    """

    predictor = MainPredictor(user_need=user_need)

    output_folder = os.path.join(base_path, f"image_{image_no}")
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "prediction_results.txt")
    final_characters = []

    with open(output_file, "w", encoding="utf-8") as f:
        for char_num in range(1, total_chars + 1):
            img_path = os.path.join(output_folder, f"{image_no}_image_character_{char_num}.png")
            print(f"🔍 Predicting: {img_path}")

            if not os.path.exists(img_path):
                f.write(f"{image_no}_{char_num}: ❌ ERROR - File not found\n")
                final_characters.append("□")
                continue

            try:
                result = predictor.predict(img_path)

                if not isinstance(result, dict):
                    raise ValueError("predict() did not return a dict")

                final_char = result.get("Final")

                if isinstance(final_char, str) and final_char.strip():
                    final_characters.append(final_char)
                else:
                    final_characters.append("□")

                f.write(f"{image_no}_{char_num}:\n")
                f.write(f"  Random Forest : {result.get('Random Forest', 'N/A')}\n")
                f.write(f"  Extra Trees   : {result.get('Extra Trees', 'N/A')}\n")
                f.write(f"  XGBoost       : {result.get('XGBoost', 'N/A')}\n")
                f.write(f"  Final         : {final_char or '□'}\n\n")

            except Exception as e:
                f.write(f"{image_no}_{char_num}: ❌ ERROR - {str(e)}\n")
                final_characters.append("□")

    final_sequence = ' '.join(final_characters)

    return {
        "image_no": image_no,
        "total_chars": total_chars,
        "final_sequence": final_sequence,
        "output_file": output_file,
        "output_dir": output_folder
    }

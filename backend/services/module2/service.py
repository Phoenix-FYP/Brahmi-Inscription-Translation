# backend/services/module2/service.py

from .main import process_characters

def run_module2(image_no: int, total_chars: int, user_need: str = "both", output_base: str = "./results/module2") -> dict:
    """
    Wrapper for running character prediction for Module 2.
    """
    return process_characters(image_no=image_no, total_chars=total_chars, user_need=user_need, base_path=output_base)

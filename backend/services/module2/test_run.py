# backend/services/module2/test_run.py

from .service import run_module2

if __name__ == "__main__":
    result = run_module2(image_no=28, total_chars=12)
    print("\n✅ Module 2 Result:")
    print(result)

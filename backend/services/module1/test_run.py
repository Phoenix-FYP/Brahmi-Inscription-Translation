from .service import run_module1
import os

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.abspath(os.path.join(BASE_DIR, '../../../data/module-1/images/12.jpg'))
    result = run_module1(image_path=image_path, image_no=12)
    print("\nResult Summary:")
    print(result)

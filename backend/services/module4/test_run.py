from services.module4.service import run_module4

brhami_input = ['බමණ', 'උතර', 'පුත', 'ගුතහ', 'ලෙණෙ', 'ශගශ']
output = run_module4(brhami_input)

print("\n Module 4 Output:")
for k, v in output.items():
    print(f"{k}: {v}")

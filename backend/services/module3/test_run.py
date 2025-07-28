from .service import run_module3

sample_text = "පමමකශිවහලෙණෙශගශ"  # Or the output from Module 2
result = run_module3(sample_text)

print("\nFinal Segmented Output:")
print("Segmented:", " ".join(result["best"]["words"]))
if result["needs_correction"]:
    print("Corrected:", result["corrected"])

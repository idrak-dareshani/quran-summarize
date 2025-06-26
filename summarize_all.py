import os
from src.summarizer import summarize_file

input_dir = "data/json"
output_dir = "data/summaries"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all JSON files in the input directory
for file in os.listdir(input_dir):
    if file.endswith(".json"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".json", ".txt"))
        print(f"⏳ Processing: {file}")
        summarize_file(input_path, output_path)
        print(f"✅ Summary saved to: {output_path}")

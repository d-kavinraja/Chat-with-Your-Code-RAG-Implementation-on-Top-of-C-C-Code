import os
import json
from pathlib import Path
from tqdm import tqdm

# Set the path to your LPrint source folder (adjust this if needed)
LPRINT_SOURCE_DIR = "../lprint"  # or "LPRINT" if it's in same folder
OUTPUT_JSON = "lprint_chunks.json"

# Allowed file types
ALLOWED_EXTENSIONS = [".c", ".h"]

# Chunking configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def collect_chunks():
    data = []
    for root, _, files in os.walk(LPRINT_SOURCE_DIR):
        for filename in tqdm(files, desc="Chunking LPrint files"):
            ext = Path(filename).suffix
            if ext in ALLOWED_EXTENSIONS:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, LPRINT_SOURCE_DIR)
                content = read_file_content(full_path)
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    data.append({
                        "id": f"{rel_path}-{i}",
                        "source": rel_path,
                        "content": chunk
                    })
    return data

def main():
    chunks = collect_chunks()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nDone! {len(chunks)} chunks saved to '{OUTPUT_JSON}'")

if __name__ == "__main__":
    main()

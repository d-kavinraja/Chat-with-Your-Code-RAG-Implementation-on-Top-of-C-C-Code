# import os
# import json
# from pathlib import Path
# from tqdm import tqdm

# # Set the path to your LPrint source folder (adjust this if needed)
# LPRINT_SOURCE_DIR = "../lprint"  # or "LPRINT" if it's in same folder
# OUTPUT_JSON = "lprint_chunks.json"

# # Allowed file types
# ALLOWED_EXTENSIONS = [".c", ".h"]

# # Chunking configuration
# CHUNK_SIZE = 300
# CHUNK_OVERLAP = 50

# def read_file_content(file_path):
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         return f.read()

# def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         if chunk:
#             chunks.append(chunk)
#     return chunks

# def collect_chunks():
#     data = []
#     for root, _, files in os.walk(LPRINT_SOURCE_DIR):
#         for filename in tqdm(files, desc="Chunking LPrint files"):
#             ext = Path(filename).suffix
#             if ext in ALLOWED_EXTENSIONS:
#                 full_path = os.path.join(root, filename)
#                 rel_path = os.path.relpath(full_path, LPRINT_SOURCE_DIR)
#                 content = read_file_content(full_path)
#                 chunks = chunk_text(content)
#                 for i, chunk in enumerate(chunks):
#                     data.append({
#                         "id": f"{rel_path}-{i}",
#                         "source": rel_path,
#                         "content": chunk
#                     })
#     return data

# def main():
#     chunks = collect_chunks()
#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(chunks, f, indent=2, ensure_ascii=False)
#     print(f"\nDone! {len(chunks)} chunks saved to '{OUTPUT_JSON}'")

# if __name__ == "__main__":
#     main()

import os
import json
from pathlib import Path
from tqdm import tqdm

# Multiple source directories - add your own paths here
SOURCE_DIRECTORIES = [
    "../lprint",        # Original LPrint code
    "./your_cpp_code",  # Your additional C++ code
    "./uploaded_code"   # Directory for uploaded files
]

OUTPUT_JSON = "combined_chunks.json"
ALLOWED_EXTENSIONS = [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]

# Chunking configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def read_file_content(file_path):
    """Read file content with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def extract_functions(content):
    """Extract function names from C/C++ code"""
    import re
    pattern = re.compile(r'\b(\w+)\s*\([^)]*\)\s*\{')
    return pattern.findall(content)

def collect_chunks_from_multiple_dirs():
    """Process all directories and collect chunks"""
    data = []
    total_files = 0
    
    for source_dir in SOURCE_DIRECTORIES:
        if not os.path.exists(source_dir):
            print(f"⚠️  Warning: Directory {source_dir} does not exist, skipping...")
            continue
            
        print(f"📁 Processing directory: {source_dir}")
        
        # Count files first
        file_count = 0
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if Path(filename).suffix.lower() in ALLOWED_EXTENSIONS:
                    file_count += 1
        
        if file_count == 0:
            print(f"   No C/C++ files found in {source_dir}")
            continue
            
        # Process files with progress bar
        with tqdm(total=file_count, desc=f"Processing {os.path.basename(source_dir)}") as pbar:
            for root, _, files in os.walk(source_dir):
                for filename in files:
                    ext = Path(filename).suffix.lower()
                    if ext in ALLOWED_EXTENSIONS:
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, source_dir)
                        
                        try:
                            content = read_file_content(full_path)
                            if content:
                                chunks = chunk_text(content)
                                functions = extract_functions(content)
                                
                                for i, chunk in enumerate(chunks):
                                    data.append({
                                        "id": f"{os.path.basename(source_dir)}-{rel_path}-{i}",
                                        "source": f"{os.path.basename(source_dir)}/{rel_path}",
                                        "content": chunk,
                                        "project": os.path.basename(source_dir),
                                        "file_type": ext,
                                        "functions": functions if i == 0 else [],  # Add functions to first chunk only
                                        "chunk_index": i
                                    })
                                total_files += 1
                        except Exception as e:
                            print(f"   ❌ Error processing {full_path}: {e}")
                        
                        pbar.update(1)
    
    return data, total_files

def main():
    print("🚀 Starting multi-directory C/C++ code chunking...")
    
    chunks, total_files = collect_chunks_from_multiple_dirs()
    
    if not chunks:
        print("❌ No chunks generated! Please check your source directories.")
        return
    
    # Save chunks to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    projects = set(chunk["project"] for chunk in chunks)
    file_types = {}
    for chunk in chunks:
        ext = chunk["file_type"]
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"\n✅ Done! Generated {len(chunks)} chunks from {total_files} files")
    print(f"📊 Projects processed: {', '.join(projects)}")
    print(f"📋 File types: {dict(file_types)}")
    print(f"💾 Saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

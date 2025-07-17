import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load the chunks JSON from absolute or relative path
with open("lprint_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Convert chunks to LangChain Documents
docs = [Document(page_content=chunk["content"], metadata={"source": chunk["source"]}) for chunk in chunks]

# Create the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
db = FAISS.from_documents(docs, embedding_model)

# Save the index to 'faiss_index/' directory
index_dir = "faiss_index"
os.makedirs(index_dir, exist_ok=True)  # create folder if not exists
db.save_local(index_dir)

print(f"FAISS index saved to: {index_dir}")

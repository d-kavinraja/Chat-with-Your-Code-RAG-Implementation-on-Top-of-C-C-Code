# # import json
# # import os
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_core.documents import Document

# # # Load the chunks JSON from absolute or relative path
# # with open("lprint_chunks.json", "r", encoding="utf-8") as f:
# #     chunks = json.load(f)

# # # Convert chunks to LangChain Documents
# # docs = [Document(page_content=chunk["content"], metadata={"source": chunk["source"]}) for chunk in chunks]

# # # Create the embedding model
# # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # # Create FAISS vector store
# # db = FAISS.from_documents(docs, embedding_model)

# # # Save the index to 'faiss_index/' directory
# # index_dir = "faiss_index"
# # os.makedirs(index_dir, exist_ok=True)  # create folder if not exists
# # db.save_local(index_dir)

# # print(f"FAISS index saved to: {index_dir}")
# import json
# import os
# from langchain_community.vectorstores import FAISS, Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.documents import Document
# from tqdm import tqdm

# # Configuration
# CHUNKS_FILE = "combined_chunks.json"
# FAISS_INDEX_DIR = "faiss_index"
# CHROMA_DB_DIR = "./chroma"

# def load_chunks():
#     """Load chunks from JSON file"""
#     try:
#         with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
#             chunks = json.load(f)
#         print(f"📖 Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
#         return chunks
#     except FileNotFoundError:
#         print(f"❌ File {CHUNKS_FILE} not found. Run chunk_lprint_code.py first.")
#         return []
#     except Exception as e:
#         print(f"❌ Error loading chunks: {e}")
#         return []

# def create_documents(chunks):
#     """Convert chunks to LangChain Documents with enhanced metadata"""
#     docs = []
#     for chunk in tqdm(chunks, desc="Creating documents"):
#         doc = Document(
#             page_content=chunk["content"],
#             metadata={
#                 "source": chunk["source"],
#                 "project": chunk["project"],
#                 "file_type": chunk["file_type"],
#                 "chunk_id": chunk["id"],
#                 "chunk_index": chunk["chunk_index"],
#                 "functions": chunk.get("functions", [])
#             }
#         )
#         docs.append(doc)
#     return docs

# def create_embeddings():
#     """Create embedding model"""
#     print("🔧 Initializing embedding model...")
#     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def create_faiss_index(docs, embeddings):
#     """Create and save FAISS index"""
#     print("🏗️  Creating FAISS vector store...")
#     db = FAISS.from_documents(docs, embeddings)
    
#     # Create directory if it doesn't exist
#     os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
#     # Save the index
#     db.save_local(FAISS_INDEX_DIR)
#     print(f"💾 FAISS index saved to: {FAISS_INDEX_DIR}")
#     return db

# def create_chroma_index(docs, embeddings):
#     """Create and save Chroma index"""
#     print("🏗️  Creating Chroma vector store...")
    
#     # Remove existing directory if it exists
#     if os.path.exists(CHROMA_DB_DIR):
#         import shutil
#         shutil.rmtree(CHROMA_DB_DIR)
    
#     # Create new Chroma DB
#     db = Chroma.from_documents(
#         documents=docs,
#         embedding=embeddings,
#         persist_directory=CHROMA_DB_DIR
#     )
    
#     # Persist the database
#     db.persist()
#     print(f"💾 Chroma database saved to: {CHROMA_DB_DIR}")
#     return db

# def main():
#     print("🚀 Starting embedding generation...")
    
#     # Load chunks
#     chunks = load_chunks()
#     if not chunks:
#         return
    
#     # Create documents
#     docs = create_documents(chunks)
    
#     # Create embeddings
#     embeddings = create_embeddings()
    
#     # Create both FAISS and Chroma indexes
#     print("\n📊 Creating vector stores...")
    
#     try:
#         # FAISS index
#         faiss_db = create_faiss_index(docs, embeddings)
#         print(f"✅ FAISS index created with {len(docs)} documents")
#     except Exception as e:
#         print(f"❌ Error creating FAISS index: {e}")
    
#     try:
#         # Chroma index
#         chroma_db = create_chroma_index(docs, embeddings)
#         print(f"✅ Chroma index created with {len(docs)} documents")
#     except Exception as e:
#         print(f"❌ Error creating Chroma index: {e}")
    
#     # Print statistics
#     projects = set(doc.metadata["project"] for doc in docs)
#     file_types = {}
#     for doc in docs:
#         ext = doc.metadata["file_type"]
#         file_types[ext] = file_types.get(ext, 0) + 1
    
#     print(f"\n📊 Embedding Statistics:")
#     print(f"   Total documents: {len(docs)}")
#     print(f"   Projects: {', '.join(projects)}")
#     print(f"   File types: {dict(file_types)}")
#     print(f"   Embedding model: all-MiniLM-L6-v2")

# if __name__ == "__main__":
#     main()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Configuration
USE_CHROMA = True  # Set to False to use FAISS
CHROMA_DB_DIR = "./chroma"
FAISS_INDEX_DIR = "faiss_index"

def load_vector_store():
    """Load vector store (Chroma or FAISS)"""
    print("🔧 Loading vector store...")
    
    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        if USE_CHROMA and os.path.exists(CHROMA_DB_DIR):
            db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embedding_model
            )
            print("✅ Loaded Chroma vector store")
        elif os.path.exists(FAISS_INDEX_DIR):
            db = FAISS.load_local(
                FAISS_INDEX_DIR, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            print("✅ Loaded FAISS vector store")
        else:
            print("❌ No vector store found! Run embed_chunks.py first.")
            return None
        
        return db
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None

def create_qa_chain(db):
    """Create RAG QA chain"""
    print("🔧 Setting up QA chain...")
    
    # Create retriever
    retriever = db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    )
    
    # Enhanced prompt template
    prompt_template = """
    You are a helpful assistant specialized in C/C++ source code analysis.
    Use the following retrieved code snippets to answer the user query as accurately as possible.

    Code Context:
    {context}

    Question: {question}

    Instructions:
    - Explain the purpose and functionality of relevant code
    - Provide specific examples from the code when possible
    - Mention source file names for reference
    - If discussing functions, explain their parameters and return values

    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Use TinyLlama model
    llm = ChatOllama(model="tinyllama")
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✅ QA chain ready!")
    return qa_chain

def print_sources(docs):
    """Print source documents information"""
    if not docs:
        return
    
    print("\n📋 Source References:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        project = doc.metadata.get('project', 'Unknown')
        print(f"   {i}. {project}/{source}")

def main():
    print("🤖 Enhanced C/C++ Code Assistant")
    print("=" * 50)
    
    # Load vector store
    db = load_vector_store()
    if not db:
        return
    
    # Create QA chain
    qa_chain = create_qa_chain(db)
    
    print("\n💬 Ask questions about your C/C++ codebase!")
    print("   Type 'help' for examples, 'quit' or 'exit' to stop")
    print("-" * 50)
    
    # Chat loop
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
                
            if query.lower() == "help":
                print("\n💡 Example questions:")
                print("   • How does the Brother printer driver work?")
                print("   • What functions handle error checking?")
                print("   • Explain the dithering algorithm")
                print("   • Show me memory management in the code")
                print("   • What are the supported media types?")
                continue
            
            print("\n🤔 Analyzing code...")
            result = qa_chain.invoke({"query": query})
            
            print(f"\n🤖 Answer:")
            print(f"{result['result']}")
            
            # Show sources
            print_sources(result.get('source_documents', []))
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please ensure Ollama is running with the tinyllama model")

if __name__ == "__main__":
    main()

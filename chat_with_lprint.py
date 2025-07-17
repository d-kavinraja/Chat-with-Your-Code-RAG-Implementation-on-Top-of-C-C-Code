from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Embedding model (same)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_store", embedding_model, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Use TinyLLaMA model instead of Mistral
llm = ChatOllama(model="tinyllama")  # Use the newly downloaded smaller model

# Create RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Chat loop
print("[Chat] Ask a question about the LPrint codebase (type 'exit' to quit):")
while True:
    query = input("\n> ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke(query)
    print("\n[Answer]:\n", result["result"])

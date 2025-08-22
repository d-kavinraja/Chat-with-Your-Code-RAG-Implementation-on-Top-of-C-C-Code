# Chat-with-Your-Code-RAG-Implementation-on-Top-of-C-C-Code

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to interact with the LPrint C/C++ codebase in natural language. By combining chunking, embeddings, and a conversational interface, it enables developers to query the code, understand its structure, and get explanations directly from an AI assistant.

🚀 Features

Code Chunking – Splits large source files into manageable text segments.

Embeddings – Generates vector representations of code chunks for semantic search.

RAG Pipeline – Retrieves relevant chunks based on user queries and augments the LLM response.

Interactive Chat – Provides a chatbot-style interface to ask questions about the repository.

Preprocessed Data – Includes stored embeddings (.npy) and JSON mappings for quick setup.

📂 Project Structure

app.py → Main application entry point

chat_with_lprint.py → Chat interface with RAG integration

chunk_lprint_code.py → Splits code into chunks

embed_chunks.py → Generates embeddings for code chunks

lprint_chunks.json / lprint_texts.json / lprint_embeddings.npy → Preprocessed dataset files

requirements.txt → Dependencies list

README.md → Documentation

LICENSE → License information

🔧 Tech Stack

Python 3

LangChain (for RAG pipeline)

ChromaDB (vector database for embeddings)

Ollama / TinyLlama (LLM for code Q&A)

📌 Use Case

This project is useful for:

Developers trying to understand large C/C++ repositories.

Students learning how to integrate RAG with codebases.

Building interactive assistants for open-source projects.

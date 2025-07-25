# Chat-with-Your-Code-RAG-Implementation-on-Top-of-C-C-Code

colab link : https://colab.research.google.com/drive/1z0xsmf7R72P6SQnD73SafuN41ksKYwqM?usp=sharing

check this link : https://github.com/P-PRIYA-VARSHA/Sasken-Project
this is what i did ..


# CodeInsight AI - RAG-Based C/C++ Code Assistant

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-yellow.svg)](https://langchain.com)

> **AI-powered code analysis platform for C/C++ codebases with intelligent Q&A, visual diagrams, and semantic search capabilities.**


## ✨ Features

### 🤖 **Intelligent Code Analysis**
- **Natural Language Queries**: Ask questions about your code in plain English
- **Context-Aware Responses**: Get detailed explanations with code examples and citations
- **Function Relationship Analysis**: Understand how functions interact within your codebase
- **Memory Management Insights**: Analyze allocation patterns and potential issues

### 📊 **Visual Code Insights**
- **Interactive Mermaid Diagrams**: Auto-generated function call graphs and architecture diagrams
- **Smart Caching System**: 90% faster diagram generation with intelligent caching
- **Multi-perspective Views**: Function relationships, structures, and file dependencies
- **Real-time Visualization**: Dynamic diagram updates based on code analysis

### ⚡ **Performance Optimized**
- **Sub-3-second Query Times**: Optimized retrieval with semantic search
- **Parallel Processing**: Multi-threaded file upload and chunking
- **Smart Chunking**: Efficient code segmentation for maximum relevance
- **Intelligent Caching**: Reduces repeated computation overhead

### 🔧 **Developer-Friendly Interface**
- **Modern Web UI**: Professional Streamlit interface with custom styling
- **Multiple Input Methods**: File upload, direct code pasting, and batch processing
- **Audio Support**: Text-to-speech for accessibility
- **Query History**: Track and revisit previous analyses
- **Function Reference**: Live sidebar with discovered functions

## 🏗️ System Architecture (Currently Working)


## 📦 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **[Ollama](https://ollama.ai/)** for local LLM hosting
- **Git** for version control
- **8GB+ RAM** (16GB recommended)

### 1. Clone the Repository
### 2. Install Python Dependencies

### 3. Set Up Ollama


### 4. Initialize the System

### 5. Launch the Application

## 🚀 Quick Start Guide

### Upload Your Codebase
1. **Launch the web interface**: `streamlit run app.py`
2. **Upload files**: Use the file uploader for `.c`, `.cpp`, `.h`, `.hpp` files
3. **Or paste code**: Directly input code snippets via the text interface
4. **Processing**: Files are automatically chunked and indexed


### Explore Visual Insights
- **Click "📊 Diagram"** to generate function relationship graphs
- **View Source References** to see exact code citations
- **Use Audio Playback** for accessibility
- **Browse Function Reference** in the sidebar

## 📁 Project Structure

CodeInsight-AI/
├── 🎯 app.py # Main Streamlit web application
├── 💬 chat_with_lprint.py # Command-line interface
├── 🔧 chunk_lprint_code.py # Code chunking and preprocessing
├── 🧠 embed_chunks.py # Embedding generation pipeline
├── 📋 requirements.txt # Python dependencies
├── 📖 README.md # Project documentation
├── 📁 chroma/ # ChromaDB vector storage
├── 📁 lprint/ # Sample C codebase (Brother printer drivers)
├── 📄 lprint_chunks.json # Processed code chunks
├── 📄 lprint_texts.json # Text processing cache
└── 📁 diagram_cache/ # Mermaid diagram cache


## ⚙️ Configuration

### Environment Variables
##### Optional: Customize performance settings
##### export CHUNK_SIZE=200 # Words per chunk
##### export CHUNK_OVERLAP=30 # Word overlap between chunks
##### export RETRIEVAL_K=3 # Number of chunks to retrieve
##### export OLLAMA_HOST=localhost:11434 # Ollama server address


### Model Configuration
##### In app.py - modify these for different models:
##### EMBEDDING_MODEL = "nomic-embed-text"
##### LLM_MODEL = "tinyllama"
##### VECTOR_DB_PATH = "./chroma"





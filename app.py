import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import streamlit.components.v1 as components
import re
import pyttsx3
import tempfile
import os
import time

st.set_page_config(page_title="CodeSense: Chat with LPrint", layout="wide", page_icon="💻")

# Hero Section with Gradient Background
st.markdown("""
<style>
.hero {
    background: linear-gradient(to right, #4facfe, #00f2fe);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}
.hero p {
    font-size: 1.1rem;
}
</style>
<div class="hero">
  <h1>💡 CodeSense: LPrint Code Assistant</h1>
  <p>Ask anything about your C/C++ codebase — from logic to memory usage to function flow.</p>
</div>
""", unsafe_allow_html=True)

def text_to_speech(text):
    engine = pyttsx3.init()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_path = temp_file.name
    temp_file.close()
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    with open(temp_path, 'rb') as f:
        audio_bytes = f.read()
    try:
        os.remove(temp_path)
    except PermissionError:
        pass
    return audio_bytes

def extract_function_calls(code: str) -> list:
    pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')
    blacklist = ['if', 'for', 'while', 'switch', 'return', 'sizeof']
    calls = pattern.findall(code)
    return sorted(set([call for call in calls if call not in blacklist]))

def extract_function_names(chunks):
    function_names = []
    pattern = re.compile(r'\b\w+\s+\**(\w+)\s*\([^)]*\)\s*\{')
    for chunk in chunks:
        matches = pattern.findall(chunk.page_content)
        function_names.extend(matches)
    return sorted(set(function_names))

@st.cache_resource
def load_rag_components():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="./chroma", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    prompt_template = """
You are CodeSense, an expert C/C++ code assistant. Your job is to answer questions using the provided code snippets.

Each snippet may contain parts of functions, comments, or logic blocks. Use all available information to:
- Understand the purpose of functions and variables
- Trace the flow of logic or function calls
- Explain memory handling, I/O, and other system-level behavior

IMPORTANT:
- Be accurate and concise
- If something is unclear or not present in the snippets, say "Not enough information"
- Use the function or variable names as they appear

Code Snippets:
=====================
{context}
=====================

User Question: {question}

Detailed Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = Ollama(
    model="tinyllama",
    temperature=0.5,     # Lower = more deterministic
    top_p=0.5,          # Controls diversity
    repeat_penalty=1.1,  # Optional: discourage repetition
        )

    return llm, retriever, prompt, vectordb

llm, retriever, prompt, vectordb = load_rag_components()
chunks = vectordb.get()["documents"]

with st.sidebar:
    st.markdown("""
        <style>
        .sidebar-section {
            margin-bottom: 1.5rem;
        }
        .sidebar-section h3 {
            margin-top: 1rem;
            font-size: 1.1rem;
            color: #0d47a1;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'><h3>📁 Functions in Codebase</h3>", unsafe_allow_html=True)
    for fname in extract_function_names(chunks):
        st.markdown(f"- `{fname}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><h3>🔍 Sample Questions</h3>", unsafe_allow_html=True)
    st.markdown("- What does `lprintAddPrinter` do?")
    st.markdown("- How does the logging mechanism work?")
    st.markdown("- Which function handles print jobs?")
    st.markdown("- Is memory allocation handled safely?")
    st.markdown("- Describe the flow of `main()` function")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><h3>ℹ️ Powered By</h3>", unsafe_allow_html=True)
    st.markdown("- TinyLLaMA via Ollama")
    st.markdown("- ChromaDB for Retrieval")
    st.markdown("- Streamlit for UI")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# User Input
# ------------------------------
query = st.text_input("🔍 Ask a question", placeholder="e.g., What does lprintAddPrinter do?")

if "query_log" not in st.session_state:
    st.session_state.query_log = []

# ------------------------------
# Query Processing
# ------------------------------
if query:
    with st.spinner("⚙️ Generating answer..."):
        full_start = time.time()

        # Step 1: Document Retrieval
        retrieval_start = time.time()
        docs = retriever.get_relevant_documents(query)
        retrieval_end = time.time()

        # Step 2: LLM Inference
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = prompt.format(context=context, question=query)
        generation_start = time.time()
        answer = llm.invoke(formatted_prompt)
        generation_end = time.time()

        full_end = time.time()

        # Timings
        retrieval_time = retrieval_end - retrieval_start
        generation_time = generation_end - generation_start
        total_time = full_end - full_start

        # ------------------------------
        # Display Answer
        # ------------------------------
        st.markdown("## 🧠 Answer")
        st.success(answer)

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Retrieval Time", f"{retrieval_time:.2f}s")
        col2.metric("🤖 Generation Time", f"{generation_time:.2f}s")
        col3.metric("⏱️ Total Time", f"{total_time:.2f}s")

        with st.expander("🔊 Listen to Answer"):
            audio_data = text_to_speech(answer)
            st.audio(audio_data, format='audio/mp3')

        # ------------------------------
        # Source Code Chunks
        # ------------------------------
        if docs:
            st.markdown("## 🧩 Source Code Chunks Used")
            for i, doc in enumerate(docs):
                with st.expander(f"Chunk {i+1}: {doc.metadata.get('file')}"):
                    st.code(doc.page_content, language="c")

        # ------------------------------
        # Modern Vertical Flowchart from Answer
        # ------------------------------
        st.markdown("## 🔄 Function Diagram")

        def extract_steps_from_answer(text):
            lines = re.split(r'\.|\n|\->|→|⇒', text)
            steps = [line.strip(" -•\n") for line in lines if len(line.strip()) > 5]
            return steps[:6]

        steps = extract_steps_from_answer(answer)

        if steps:
            flowchart_html = """
            <style>
            .flow-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
                padding-top: 20px;
            }
            .step-box {
                background-color: #e3f2fd;
                padding: 12px 20px;
                border-radius: 10px;
                font-weight: 600;
                color: #0d47a1;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                text-align: center;
                width: max-content;
                max-width: 80%;
                border-left: 6px solid #2196f3;
            }
            .start, .end {
                background-color: #c8e6c9;
                border-left-color: #388e3c;
            }
            .arrow-down {
                font-size: 28px;
                color: #444;
            }
            </style>
            <div class='flow-container'>
            """

            for i, step in enumerate(steps):
                shape_class = "start" if i == 0 else "end" if i == len(steps)-1 else ""
                flowchart_html += f"<div class='step-box {shape_class}'>{step}</div>"
                if i < len(steps) - 1:
                    flowchart_html += "<div class='arrow-down'>⬇️</div>"

            flowchart_html += "</div>"

            st.components.v1.html(flowchart_html, height=150 + 100 * len(steps), scrolling=True)
        else:
            st.info("No clear steps could be extracted for flowchart.")

        # ------------------------------
        # Answer Rating
        # ------------------------------
        with st.expander("⭐ Rate this answer"):
            rating = st.radio("Was this helpful?", ["Excellent", "Good", "Average", "Poor"], key=query)
            st.session_state.query_log.append((query, answer, rating))

# ------------------------------
# Query Log
# ------------------------------
with st.expander("📜 Query History"):
    for q, a, r in reversed(st.session_state.query_log):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown(f"**🗳️ User Rating:** `{r}`")
        st.markdown("---")
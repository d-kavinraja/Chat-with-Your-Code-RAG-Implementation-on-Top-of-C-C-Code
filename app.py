import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit.components.v1 as components
import re
import pyttsx3
import tempfile
import os

# Function to extract function names from code chunks
def extract_function_names(chunks):
    function_names = []
    pattern = re.compile(r'\b\w+\s+\**(\w+)\s*\([^)]*\)\s*\{')
    for chunk in chunks:
        matches = pattern.findall(chunk.page_content)
        function_names.extend(matches)
    return sorted(set(function_names))

# Text-to-speech function
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

# Streamlit app config
st.set_page_config(page_title="RAG-Based Code Assistant", layout="wide")
st.title("Chat with Your C/C++ Codebase")

# Sidebar
with st.sidebar:
    st.header("Code Assistant Settings")
    st.markdown("""
    This assistant uses TinyLlama and ChromaDB to answer your queries about the LPrint codebase.
    You can ask questions like:
    - What does this function do?
    - Where is this variable used?
    - Explain memory management in buffer.c
    """)
    st.divider()
    st.write("Model: tinyllama via Ollama")
    st.write("Vector DB: Chroma (./chroma)")
    st.write("Flowchart: Mermaid diagram")

# Load RAG chain
@st.cache_resource
def load_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="./chroma", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """
    You are a helpful assistant specialized in C/C++ source code. Use the following retrieved code snippets to answer the user query as accurately as possible.

    ---------------------
    {context}
    ---------------------
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = Ollama(model="tinyllama")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain, vectordb

qa_chain, vectordb = load_rag_chain()

# Extract function names from chunks
chunks = vectordb.get()["documents"]
with st.sidebar:
    st.subheader("Function Reference")
    functions = extract_function_names(chunks)
    for fname in functions:
        st.markdown(f"- {fname}")

# User input
query = st.text_input("Ask a question about the codebase", placeholder="e.g., What does the lprintAddPrinter function do?")

# Query log initialization
if "query_log" not in st.session_state:
    st.session_state.query_log = []

# Process query
if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        answer = result["result"]
        docs = result["source_documents"]

        st.markdown("### Answer")
        st.markdown(answer)

        # Play audio
        audio_data = text_to_speech(answer)
        st.audio(audio_data, format='audio/mp3')

        # Show source code chunks
        if answer and docs:
          st.markdown("### Source Code Chunks Used")
          for i, doc in enumerate(docs):
              st.markdown(f"**Chunk {i+1}-{doc.metadata.get('file')}**")
              st.code(doc.page_content, language="c")
              
        # Add to query log
        st.session_state.query_log.append((query, answer))
# Mermaid diagram logic (static mapping for now)
function_call_map = {
    "lprintAddPrinter": ["lprintIsPrinterInUse", "lprintLog", "lprintSavePrinters"],
    "lprintDeletePrinter": ["lprintLog", "lprintSavePrinters"],
    "lprintGetPrinter": [],
}

# Show Mermaid diagram if query matches a function
for func in function_call_map:
    if func in query:
        callees = function_call_map[func]
        st.markdown("### Function Call Diagram (Mermaid)")

        diagram = f"graph TD\n    {func}"
        for callee in callees:
            diagram += f" --> {callee}"

        components.html(f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <div class="mermaid">
        {diagram}
        </div>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """, height=400)

        break  # Stop checking after the first match
   

# Query log viewer
with st.expander("Query Log"):
    for q, a in reversed(st.session_state.query_log):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
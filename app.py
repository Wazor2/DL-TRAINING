import os
import torch
import streamlit as st
import ollama

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
#  GPU check — crash early with a clear message
#  if CUDA is not available instead of silently
#  falling back to CPU.
# ─────────────────────────────────────────────
if not torch.cuda.is_available():
    st.error(
        "❌ **No CUDA GPU detected.**\n\n"
        "Make sure you have:\n"
        "1. A CUDA-capable NVIDIA GPU\n"
        "2. The CUDA-enabled build of PyTorch installed\n\n"
        "**Run these two commands in your terminal:**\n"
        "```bash\n"
        "pip uninstall torch torchvision -y\n"
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        "```\n\n"
        "Then restart Streamlit."
    )
    st.stop()

# Force every PyTorch tensor op to land on GPU 0
DEVICE      = "cuda"
GPU_NAME    = torch.cuda.get_device_name(0)
VRAM_TOTAL  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GiB

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="C++ RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 C++ RAG Tutorial Chatbot")
st.caption(f"LangChain · FAISS · HuggingFace Embeddings · Ollama LLM · ⚡ {GPU_NAME}")

# ─────────────────────────────────────────────
#  Step 1 – Load & Split  (Cell 0 + Cell 1)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="📄 Loading CPP.txt and embedding on GPU…")
def build_vector_store(chunk_size: int, chunk_overlap: int):
    """
    Mirrors the notebook pipeline — everything runs on CUDA:
      Cell 0 – TextLoader
      Cell 1 – RecursiveCharacterTextSplitter
      Cell 2 – HuggingFaceEmbeddings on CUDA (all-MiniLM-L6-v2)
      Cell 4 – FAISS.from_documents
    """
    file_path = "CPP.txt"
    if not os.path.exists(file_path):
        st.error(f"❌ '{file_path}' not found. Place CPP.txt next to app.py.")
        st.stop()

    # Cell 0 – load
    loader   = TextLoader(file_path, encoding="utf-8")
    raw_docs = loader.load()

    # Cell 1 – split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)

    # Cell 2/3 – embeddings forced onto CUDA
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={
            "device": DEVICE,           # "cuda" — always
        },
        encode_kwargs={
            "batch_size": 64,           # larger batches → faster on GPU
            "normalize_embeddings": True,
        },
    )

    # Cell 4 – FAISS index (built from GPU-computed embeddings)
    db = FAISS.from_documents(chunks, embeddings)
    return db, len(chunks)

# ─────────────────────────────────────────────
#  Sidebar – configuration
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

chunk_size    = st.sidebar.slider("Chunk Size",    100, 1000, 500, 50)
chunk_overlap = st.sidebar.slider("Chunk Overlap",   0,  300, 200, 25)
top_k         = st.sidebar.slider("Top-K Chunks",    1,    8,   3)

st.sidebar.markdown("---")

# Dynamic Ollama model list
try:
    installed = [m.model for m in ollama.list().get("models", [])]
except Exception:
    installed = ["qwen3:8b", "gemma-local:latest"]

model_options = list(dict.fromkeys(installed)) if installed else ["qwen3:8b", "gemma-local:latest"]
# Default to qwen3:8b if available, otherwise first option
default_idx = model_options.index("qwen3:8b") if "qwen3:8b" in model_options else 0
selected_model = st.sidebar.selectbox("🧠 Ollama Model", model_options, index=default_idx)

# Build (or reuse) the vector store
vector_db, chunk_count = build_vector_store(chunk_size, chunk_overlap)

# ── GPU stats (live) ──────────────────────────
vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
vram_res  = torch.cuda.memory_reserved(0)  / (1024 ** 3)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ GPU Acceleration")
st.sidebar.success(f"**{GPU_NAME}**")
st.sidebar.markdown(
    f"| | |\n|---|---|\n"
    f"| **VRAM Total** | `{VRAM_TOTAL:.1f} GiB` |\n"
    f"| **VRAM Used**  | `{vram_used:.2f} GiB` |\n"
    f"| **VRAM Reserved** | `{vram_res:.2f} GiB` |\n"
    f"| **Embeddings** | CUDA ✅ |\n"
    f"| **Ollama LLM** | GPU (auto) ✅ |\n"
    f"| **Document**   | `CPP.txt` ({chunk_count} chunks) |"
)

# ─────────────────────────────────────────────
#  RAG prompt  (mirrors the Q&A intent of the notebook)
# ─────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert C++ programming tutor. "
        "Use ONLY the retrieved context below to answer the question. "
        "If the answer is in the context, explain it clearly with code examples where useful. "
        "If it is not in the context, say so and answer from general knowledge.",
    ),
    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{question}",
    ),
])

# ─────────────────────────────────────────────
#  Chat history
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander("🔍 Retrieved context chunks"):
                for idx, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {idx}:**\n```text\n{chunk}\n```")

# ─────────────────────────────────────────────
#  Cell 5 equivalent – similarity search + LLM
# ─────────────────────────────────────────────
query = st.chat_input("Ask anything about C++  (e.g. What is a virtual function?)")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # --- Similarity search (Cell 5) ---
    with st.spinner("🔎 Searching FAISS knowledge base…"):
        retrieved = vector_db.similarity_search(query, k=top_k)
        context   = "\n\n---\n\n".join(doc.page_content for doc in retrieved)
        chunks    = [doc.page_content for doc in retrieved]

    # --- LLM generation (streamed) ---
    with st.chat_message("assistant"):
        try:
            llm   = OllamaLLM(model=selected_model)
            chain = PROMPT | llm | StrOutputParser()

            # Stream token-by-token into the chat bubble
            answer_box = st.empty()
            full_answer = ""
            with st.spinner(f"💬 Generating with `{selected_model}` on GPU…"):
                for token in chain.stream({"context": context, "question": query}):
                    full_answer += token
                    answer_box.markdown(full_answer + "▌")
            answer_box.markdown(full_answer)

            with st.expander("🔍 Retrieved context chunks"):
                for idx, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {idx}:**\n```text\n{chunk}\n```")

            # Live VRAM after inference
            vram_after = torch.cuda.memory_allocated(0) / (1024 ** 3)
            st.caption(f"⚡ VRAM in use after inference: `{vram_after:.2f} GiB` / `{VRAM_TOTAL:.1f} GiB`")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "chunks": chunks,
            })

        except Exception as exc:
            err = str(exc)
            if "404" in err or "not found" in err.lower():
                st.error(
                    f"⚠️ Model `{selected_model}` is not available in Ollama.\n\n"
                    "Run `ollama pull <model-name>` in a terminal, then refresh."
                )
            else:
                st.error(f"An error occurred: {err}")

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>RAG pipeline: **TextLoader → RecursiveCharacterTextSplitter → "
    "HuggingFace Embeddings → FAISS → Ollama LLM**</small>",
    unsafe_allow_html=True,
)

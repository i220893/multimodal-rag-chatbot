import streamlit as st
from src.rag.rag_generator import RAGGenerator
from src.retrieval.retriever import Retriever
from src.rag.prompt_templates import ZERO_SHOT_TEMPLATE, FEW_SHOT_TEMPLATE, COT_TEMPLATE
from src.llm.llm_client import call_ollama
import tempfile
import os

st.set_page_config(page_title="Multimodal RAG System", layout="wide")
st.title("📘 Multimodal RAG Chatbot (PDFs + Images)")

rag = RAGGenerator(k=20)
retriever = rag.retriever

prompt_mode = st.selectbox(
    "Prompting Strategy:",
    ["Zero-shot", "Few-shot", "Chain-of-Thought"]
)

query = st.text_input("Ask a question about the PDFs:")
uploaded_image = st.file_uploader("Or upload an image for visual search:", type=["png", "jpg", "jpeg"])

if st.button("Run RAG Query"):
    if query.strip() == "" and uploaded_image is None:
        st.error("Please enter a query or upload an image.")
    else:
        # 1. Retrieval
        if uploaded_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_image.getvalue())
                tmp_path = tmp.name

            retrieved = retriever.search_image(tmp_path, k=20)
            os.remove(tmp_path)
        else:
            retrieved = retriever.search_text(query, k=20)

        # 2. Build context
        context_str = rag.build_context(retrieved)

        # 3. Choose prompting template
        if prompt_mode == "Zero-shot":
            final_prompt = ZERO_SHOT_TEMPLATE.format(context=context_str, question=query)
        elif prompt_mode == "Few-shot":
            final_prompt = FEW_SHOT_TEMPLATE.format(context=context_str, question=query)
        else:
            final_prompt = COT_TEMPLATE.format(context=context_str, question=query)

        # 4. Generate answer
        with st.spinner("Generating answer..."):
            answer = call_ollama(final_prompt)

        # Display results
        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("🔎 Retrieved Context Chunks")
        st.write(context_str)

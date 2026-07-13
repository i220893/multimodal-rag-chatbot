# Multimodal RAG Chatbot (PDFs + Images)

## Overview

This project is a **Multimodal Retrieval-Augmented Generation (RAG) Chatbot** designed to answer questions based on the content of PDF documents and images. It leverages a local Large Language Model (LLM) via **Ollama** and uses **ChromaDB** for efficient vector retrieval.

The system supports:
- **Text Retrieval**: Search within PDF content.
- **Visual Search**: Upload an image to find relevant context.
- **Advanced Prompting**: Choose between **Zero-shot**, **Few-shot**, and **Chain-of-Thought** prompting strategies.

## Features

- 📄 **PDF Processing**: Extracts text and metadata from PDFs.
- 🖼️ **Image Support**: Handles image queries and retrieval.
- 🧠 **Local LLM**: Uses **Ollama** (default model: `mistral:7b-instruct-q4_K_M`) for privacy and local execution.
- 🗄️ **Vector Database**: **ChromaDB** stores embeddings for fast similarity search.
- 🖥️ **Interactive UI**: Built with **Streamlit** for an easy-to-use interface.

## Project Structure

```
├── src/
│   ├── rag/                # RAG pipeline and prompt templates
│   ├── retrieval/          # Retrieval logic and ChromaDB interaction
│   ├── llm/                # LLM client (Ollama wrapper)
│   ├── embeddings/         # Embedding generation logic
│   ├── pdf_processing/     # PDF parsing and OCR
│   ├── chunking/           # Text chunking strategies
│   ├── evaluation/         # Evaluation metrics
│   └── ...
├── chroma_db/              # Persistent storage for ChromaDB
├── data/                   # Directory for input data (PDFs)
├── notebooks/              # Jupyter notebooks for experiments
├── streamlit_app.py        # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup & Installation

### Prerequisites

1.  **Python 3.8+** installed.
2.  **Ollama** installed and running.
    - Download from [ollama.com](https://ollama.com/).
    - Pull the required model:
      ```bash
      ollama pull mistral:7b-instruct-q4_K_M
      ```
      *(Note: You can change the model in `src/llm/llm_client.py` if needed)*

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the Application**:
    Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Interact with the Chatbot**:
    - Open your browser (usually at `http://localhost:8501`).
    - **Select Prompting Strategy**: Choose from Zero-shot, Few-shot, or Chain-of-Thought.
    - **Ask a Question**: Type your query in the text box.
    - **Upload an Image**: (Optional) Upload an image to search for visually similar content.
    - Click **"Run RAG Query"** to get an answer based on the retrieved context.

## System Workflow

The system follows a structured pipeline from raw data to final answer generation:

### 1. Data Preprocessing & Ingestion
- **PDF Parsing**: `PyMuPDF` extracts raw text and metadata (page numbers) from PDF documents.
- **Image Extraction**: `pdf2image` converts PDF pages into images, and `EasyOCR` extracts text from charts and figures to make visual data searchable.
- **Chunking**:
    - **Text**: Split into 600-character chunks with a sliding window to preserve context.
    - **Images**: Treated as standalone chunks.

### 2. Embedding & Vector Storage
- **Multimodal Embedding**: We use **CLIP (ViT-B-32)** to generate 512-dimensional embeddings for both text and image chunks. This allows cross-modal retrieval (text-to-image search).
- **Vector Database**: Embeddings are stored in **ChromaDB**, indexed for fast similarity search.

### 3. Retrieval
- **Query Embedding**: The user's question (text or image) is embedded using the same CLIP model.
- **Similarity Search**: ChromaDB retrieves the top-k (default k=5) most similar chunks based on cosine similarity.

### 4. Generation (RAG)
- **Prompt Engineering**: The retrieved context is injected into a prompt template. The system supports:
    - **Zero-shot**: Direct answering.
    - **Few-shot**: Uses examples to guide style.
    - **Chain-of-Thought**: Encourages step-by-step reasoning.
- **LLM Inference**: The prompt is sent to a local **Mistral-7B** model via **Ollama** to generate the final answer.

### 5. Evaluation
- **Metrics**: We evaluate the system using:
    - **Retrieval Hit Rate**: Accuracy of finding the correct source page.
    - **ROUGE/BLEU**: Semantic overlap with ground truth answers.
    - **Latency**: Time taken for end-to-end generation.
- **Results**: See `report_plots/` for detailed performance graphs.

## Key Components

-   **`streamlit_app.py`**: The frontend interface. Handles user input, calls the retrieval system, and displays results.
-   **`src/rag/rag_generator.py`**: Orchestrates the RAG process (Retrieval -> Context Building -> Generation).
-   **`src/retrieval/retriever.py`**: Manages interactions with ChromaDB for text and image search.
-   **`src/llm/llm_client.py`**: Wrapper for calling the local Ollama instance.

## Dependencies

Key libraries used:
-   `streamlit`: Web interface.
-   `chromadb`: Vector database.
-   `sentence-transformers`: Embeddings.
-   `pymupdf`, `pdf2image`, `easyocr`: PDF and image processing.
-   `langchain`: Framework utilities.

See `requirements.txt` for the full list.

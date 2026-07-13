import json
import chromadb

from src.embeddings.embed_text import embed_text
from src.embeddings.embed_image import embed_image


def build_chroma(chunks_path="data/chunks.json", persist_dir="chroma_db"):
    print("📦 Initializing ChromaDB (persistent mode)...")
    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(
        name="rag_multimodal",
        metadata={"hnsw:space": "cosine"}
    )

    # Load chunks
    print("📥 Loading chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"🔢 Total chunks to embed: {len(chunks)}")

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["id"]
        ctype = chunk["type"]

        # TEXT / OCR
        if ctype in ["text", "ocr"]:
            emb = embed_text(chunk["content"])
            doc = chunk["content"]

        # IMAGE
        elif ctype == "image":
            emb = embed_image(chunk["content"])
            doc = None  # image has no document text

        else:
            continue

        if emb is None:
            continue

        # Chroma metadata CANNOT contain None!
        metadata = {
            "source_pdf": str(chunk["source_pdf"]),
            "page_num": int(chunk["page_num"]),
            "type": str(ctype),
            "content": doc if doc is not None else ""
        }

        ids.append(chunk_id)
        embeddings.append(emb)
        metadatas.append(metadata)
        documents.append(doc if doc is not None else "")

        if idx % 200 == 0:
            print(f"🔄 Embedded: {idx}/{len(chunks)}")

    print("💾 Adding embeddings to Chroma in batches...")

    batch_size = 1000
    total = len(ids)

    for i in range(0, total, batch_size):
        end = i + batch_size
        print(f"📦 Adding batch {i} → {end} of {total}")

        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            documents=documents[i:end]
        )

    print("💾 DB persisted!")
    print("🎉 Vector DB build complete!")
from src.retrieval.retriever import Retriever
from src.llm.llm_client import call_ollama


class RAGGenerator:

    def __init__(self, k=5):
        self.retriever = Retriever()
        self.k = k

    def build_context(self, retrieved_chunks):
        """
        Combine top chunks into a single context string.
        """
        metadatas = retrieved_chunks["metadatas"][0]
        documents = retrieved_chunks["documents"][0]

        context_parts = []
        for meta, doc in zip(metadatas, documents):
            chunk_text = doc if isinstance(doc, str) and doc != "" else ""
            entry = f"[PDF: {meta['source_pdf']}, Page: {meta['page_num']}, Type: {meta['type']}]\n{chunk_text}\n"
            context_parts.append(entry)

        return "\n".join(context_parts)

    def generate_answer(self, user_query):
        """
        Full RAG pipeline:
        - Embed + retrieve chunks
        - Build context
        - Pass context + question to LLM
        """
        results = self.retriever.search_text(user_query, k=self.k)
        context_str = self.build_context(results)

        prompt = f"""
You are a helpful multimodal RAG assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say: "The provided documents do not contain this information."

======== CONTEXT ========
{context_str}
======== END CONTEXT ========

QUESTION:
{user_query}

Provide a clear, structured answer.
"""

        output = call_ollama(prompt)
        return output, context_str

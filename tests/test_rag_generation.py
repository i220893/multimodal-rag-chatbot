from src.rag.rag_generator import RAGGenerator

rag = RAGGenerator(k=20)

print("\n🔍 Query:")
query = "What is the role of FYP coordinators?"

answer, context = rag.generate_answer(query)

print("\n🔎 CONTEXT USED:\n", context)
print("\n🧠 ANSWER:\n", answer)

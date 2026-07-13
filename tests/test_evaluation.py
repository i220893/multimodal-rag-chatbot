from src.retrieval.retriever import Retriever
from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k, mean_average_precision
from src.evaluation.generation_metrics import compute_bleu, compute_rouge
from src.evaluation.system_metrics import measure_latency

retriever = Retriever()

query = "Summarize the financial performance in 2024"
relevant_ids = ["id-of-manually-labeled-chunk1", "id-of-manually-labeled-chunk2"]

# Measure retrieval time
(results, latency) = measure_latency(retriever.search_text, query, 5)
retrieved_ids = results["ids"][0]

print("Latency:", latency)
print("Precision@5:", precision_at_k(relevant_ids, retrieved_ids, 5))
print("Recall@5:", recall_at_k(relevant_ids, retrieved_ids, 5))
print("MAP:", mean_average_precision(relevant_ids, [retrieved_ids]))

# BLEU/ROUGE example
reference = "The company achieved strong revenue growth and stable profits."
generated = "The firm showed stable profits and steady revenue gains."

print("BLEU:", compute_bleu(reference, generated))
print("ROUGE:", compute_rouge(reference, generated))

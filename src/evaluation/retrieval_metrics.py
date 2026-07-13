import numpy as np

def precision_at_k(relevant_ids, retrieved_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = len(set(relevant_ids) & set(retrieved_k))
    return relevant_retrieved / k


def recall_at_k(relevant_ids, retrieved_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = len(set(relevant_ids) & set(retrieved_k))
    return relevant_retrieved / len(relevant_ids) if len(relevant_ids) > 0 else 0


def average_precision(relevant_ids, retrieved_ids):
    score = 0
    num_hits = 0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / len(relevant_ids) if len(relevant_ids) else 0


def mean_average_precision(relevant_ids, list_of_ranked_results):
    ap_scores = []
    for retrieved_ids in list_of_ranked_results:
        ap_scores.append(average_precision(relevant_ids, retrieved_ids))
    return np.mean(ap_scores)

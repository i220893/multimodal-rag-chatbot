import sys
import os
import pandas as pd
from tqdm import tqdm
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.rag_generator import RAGGenerator
from src.rag.prompt_templates import ZERO_SHOT_TEMPLATE, FEW_SHOT_TEMPLATE, COT_TEMPLATE
from src.llm.llm_client import call_ollama
from src.evaluation.generation_metrics import compute_bleu, compute_rouge

# Enhanced Dataset with Retrieval Ground Truth (Source PDF + Page)
TEST_DATASET = [
    {
        "question": "What is the vision of the university?",
        "reference": "To produce world-class professionals, who are responsible citizens and good human beings.",
        "expected_pdf": "1. Annual Report 2023-24.pdf",
        "expected_page": 4
    },
    {
        "question": "Who was the chief guest at the 76th Convocation?",
        "reference": "Mr. Mian Muhammad Mansha, Chairman Nishat Group, was the chief guest.",
        "expected_pdf": "1. Annual Report 2023-24.pdf",
        "expected_page": 20
    },
    {
        "question": "Which team won the Huawei ICT Innovation Global Competition 2022-23?",
        "reference": "The team comprising Minhal Zafar, Muhammad Shaheer, and Hyder Ali from FAST NUCES ISLAMABAD.",
        "expected_pdf": "1. Annual Report 2023-24.pdf",
        "expected_page": 26
    },
    {
        "question": "Describe the location of the Chiniot-Faisalabad Campus.",
        "reference": "It is situated at the junction of Faisalabad and Chiniot, spanning around 30 acres along the Faisalabad Sargodha Road.",
        "expected_pdf": "1. Annual Report 2023-24.pdf",
        "expected_page": 6
    },
    {
        "question": "What are the grading criteria for the Final Year Project?",
        "reference": "The grading criteria include proposal defense, mid-term evaluation, and final defense.",
        "expected_pdf": "3. FYP-Handbook-2023.pdf",
        "expected_page": 10 
    },
    # NEGATIVE TEST CASE
    {
        "question": "What is the recipe for chocolate cake?",
        "reference": "The provided documents do not contain this information.",
        "expected_pdf": None, 
        "expected_page": None
    }
]

def check_retrieval_hit(retrieved_chunks, expected_pdf, expected_page):
    """
    Checks if the expected PDF and Page are present in the retrieved chunks.
    Returns True if found (Hit), False otherwise.
    """
    if expected_pdf is None:
        return True # For negative tests, we don't expect a specific retrieval hit
        
    metadatas = retrieved_chunks["metadatas"][0]
    for meta in metadatas:
        if meta['source_pdf'] == expected_pdf and meta['page_num'] == expected_page:
            return True
    return False

def evaluate_pipeline():
    print("Initializing RAG System...")
    rag = RAGGenerator(k=3)
    
    results = []
    
    strategies = {
        "Zero-shot": ZERO_SHOT_TEMPLATE,
        "Few-shot": FEW_SHOT_TEMPLATE,
        "Chain-of-Thought": COT_TEMPLATE
    }

    print(f"Starting comprehensive evaluation on {len(TEST_DATASET)} queries across {len(strategies)} strategies...")
    
    for item in tqdm(TEST_DATASET, desc="Queries"):
        query = item["question"]
        reference = item["reference"]
        expected_pdf = item.get("expected_pdf")
        expected_page = item.get("expected_page")
        
        # 1. Retrieval (Performed once per query)
        retrieval_start = time.time()
        retrieved_chunks = rag.retriever.search_text(query, k=3)
        retrieval_time = time.time() - retrieval_start
        
        # Check Retrieval Hit
        is_hit = check_retrieval_hit(retrieved_chunks, expected_pdf, expected_page)
        context_str = rag.build_context(retrieved_chunks)
        
        # 2. Generation (Iterate through all strategies)
        for strategy_name, template in strategies.items():
            final_prompt = template.format(context=context_str, question=query)
            
            gen_start = time.time()
            generated_answer = call_ollama(final_prompt)
            gen_time = time.time() - gen_start
            
            # 3. Metrics
            bleu_score = compute_bleu(reference, generated_answer)
            rouge_scores = compute_rouge(reference, generated_answer)
            
            results.append({
                "Strategy": strategy_name,
                "Query": query,
                "Expected Source": f"{expected_pdf} (p{expected_page})" if expected_pdf else "None",
                "Retrieval Hit": "Yes" if is_hit else "No",
                "Latency (s)": round(retrieval_time + gen_time, 2),
                "BLEU": round(bleu_score, 4),
                "ROUGE-L": round(rouge_scores['rougeL'].fmeasure, 4),
                "Generated Answer": generated_answer[:100].replace("\n", " ") + "..." 
            })

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Summary Metrics per Strategy
    print("\nEvaluation Results Summary by Strategy:")
    summary = df.groupby("Strategy")[["Latency (s)", "ROUGE-L"]].mean()
    print(summary.to_markdown())
    
    # Save to CSV
    output_file = "evaluation_results_comprehensive.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    evaluate_pipeline()

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def compute_bleu(reference, generated):
    return sentence_bleu([reference.split()], generated.split())


def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

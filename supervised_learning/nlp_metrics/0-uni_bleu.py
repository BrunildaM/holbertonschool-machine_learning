#!/usr/bin/env python3
"""A function that calculates the unigram BLEU score for a sentence"""
from collections import Counter


def uni_bleu(references, sentence):
    """A function that calculates the unigram BLEU score for a sentence"""
    candidate_counts = Counter(sentence)
    reference_counts = Counter()

    for reference in references:
        reference_counts.update(reference)

    clip_count = sum(min(candidate_counts[ngram], reference_counts[ngram]) for ngram in candidate_counts)
    candidate_total = sum(candidate_counts.values())
    reference_total = sum(reference_counts.values())

    precision = clip_count / candidate_total if candidate_total > 0 else 0
    brevity_penalty = min(1, len(sentence) / reference_total)

    bleu_score = precision * brevity_penalty

    return bleu_score

import re
from enum import Enum
from typing import (
    List,
    Sequence,
    Tuple
)

import editdistance
import numpy as np
from numpy.typing import NDArray

from normalizer import SUPPORTED_PUNCTUATION_SET


class Metrics(Enum):
    WER = "WER"
    PER = "PER"
    BLEU = "BLEU"


class Metric:
    def calculate(self, prediction: str, reference: str) -> Tuple[int, int]:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Metrics):
        if x is Metrics.WER:
            return WordErrorRate()
        elif x is Metrics.PER:
            return PunctuationErrorRate()
        elif x is Metrics.BLEU:
            return BLEUScore()
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type `{x}`")


class WordErrorRate(Metric):
    def calculate(self, prediction: str, reference: str) -> Tuple[int, int]:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        error_count = editdistance.eval(ref_tokens, pred_tokens)
        token_count = len(ref_tokens)

        return error_count, token_count


class PunctuationErrorRate(Metric):
    """Reference: https://arxiv.org/abs/2310.02943"""

    @staticmethod
    def _get_punctuation_indices(tokens: Sequence[str], punctuation: str) -> List[int]:
        return [i for i, t in enumerate(tokens) if t in punctuation]

    @staticmethod
    def _compute_dp_matrix(reference: Sequence[str], prediction: Sequence[str]) -> NDArray:
        m, n = len(reference), len(prediction)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if reference[i - 1] == prediction[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

        return dp

    @staticmethod
    def _backtrack(
        dp: NDArray,
        reference: Sequence[str],
        prediction: Sequence[str],
        reference_punct_indices: Sequence[int],
        prediction_punct_indices: Sequence[int],
    ) -> Tuple[int, int, int, int]:
        i, j = len(reference), len(prediction)

        reference_punct_set = set(reference_punct_indices)
        prediction_punct_set = set(prediction_punct_indices)

        num_match = 0
        num_sub = 0

        alignment = {}

        while i > 0 and j > 0:
            if reference[i - 1] == prediction[j - 1]:
                alignment[i - 1] = j - 1
                i -= 1
                j -= 1
            elif dp[i - 1, j - 1] <= dp[i - 1, j] and dp[i - 1, j - 1] <= dp[i, j - 1]:
                alignment[i - 1] = j - 1
                i -= 1
                j -= 1
            elif dp[i - 1, j] <= dp[i, j - 1]:
                i -= 1
            else:
                j -= 1

        for idx in reference_punct_set:
            if idx in alignment:
                pred_idx = alignment[idx]
                if pred_idx in prediction_punct_set:
                    if reference[idx] == prediction[pred_idx]:
                        num_match += 1
                    else:
                        num_sub += 1

        num_delete = len(reference_punct_indices) - (num_sub + num_match)
        num_insert = len(prediction_punct_indices) - (num_sub + num_match)

        return num_insert, num_delete, num_sub, num_match

    def calculate(
        self, prediction: str, reference: str, punctuation: str = SUPPORTED_PUNCTUATION_SET
    ) -> Tuple[int, int]:
        pred_tokens = [token for token in re.findall(rf"[{punctuation}]|[^{punctuation}\s]+", prediction)]
        ref_tokens = [token for token in re.findall(rf"[{punctuation}]|[^{punctuation}\s]+", reference)]

        pred_punct_indices = self._get_punctuation_indices(pred_tokens, punctuation)
        ref_punct_indices = self._get_punctuation_indices(ref_tokens, punctuation)

        dp = self._compute_dp_matrix(reference=ref_tokens, prediction=pred_tokens)

        num_insert, num_delete, num_sub, num_correct = self._backtrack(
            dp=dp,
            reference=ref_tokens,
            prediction=pred_tokens,
            reference_punct_indices=ref_punct_indices,
            prediction_punct_indices=pred_punct_indices,
        )

        error_count = num_insert + num_delete + num_sub
        denom = num_insert + num_delete + num_sub + num_correct

        return error_count, denom


class BLEUScore(Metric):
    """
    BLEU (Bilingual Evaluation Understudy) score for translation quality
    Returns scaled BLEU score (0-100) as error metric
    """

    @staticmethod
    def _get_ngrams(tokens: Sequence[str], n: int) -> dict:
        """Get n-grams from token sequence"""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    @staticmethod
    def _calculate_precision(prediction_tokens: Sequence[str], reference_tokens: Sequence[str], n: int) -> Tuple[int, int]:
        """Calculate n-gram precision"""
        pred_ngrams = BLEUScore._get_ngrams(prediction_tokens, n)
        ref_ngrams = BLEUScore._get_ngrams(reference_tokens, n)

        matches = 0
        total = 0

        for ngram, count in pred_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
            total += count

        return matches, total

    @staticmethod
    def _brevity_penalty(prediction_len: int, reference_len: int) -> float:
        """Calculate brevity penalty"""
        if prediction_len >= reference_len:
            return 1.0
        return np.exp(1 - reference_len / prediction_len) if prediction_len > 0 else 0.0

    def calculate(self, prediction: str, reference: str) -> Tuple[int, int]:
        """
        Calculate BLEU score
        Returns: (error_count, total_count) where error is scaled to match WER format
        We return (100 - BLEU*100, 100) to make it compatible with error-based metrics
        """
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 100, 100  # Maximum error if either is empty

        # Calculate precision for n-grams 1-4
        precisions = []
        for n in range(1, 5):
            if len(pred_tokens) >= n and len(ref_tokens) >= n:
                matches, total = self._calculate_precision(pred_tokens, ref_tokens, n)
                if total > 0:
                    precisions.append(matches / total)
                else:
                    precisions.append(0.0)
            else:
                precisions.append(0.0)

        # Calculate geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            geo_mean = 0.0

        # Apply brevity penalty
        bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))
        bleu = bp * geo_mean

        # Convert BLEU (0-1) to error metric (0-100)
        # Higher BLEU is better, so error = 100 - (BLEU * 100)
        bleu_percent = bleu * 100
        error = 100 - bleu_percent

        # Return as integer counts for compatibility with benchmark framework
        # We'll use a scale of 1000 to maintain precision
        error_count = int(error * 10)
        total_count = 1000

        return error_count, total_count


__all__ = [
    "Metric",
    "Metrics",
]

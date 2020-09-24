import collections
import math

__all__ = ['bleu', 'perplexity']


def _count_ngram(n, hyp, refs):
    def _count(n, tokens):
        counter = collections.Counter()
        for i in range(n, len(tokens) + 1):
            counter.update([tuple(tokens[i - n:i])])
        return counter

    count_hyp = _count(n, hyp)
    count_refs = [_count(n, ref) for ref in refs]
    count_ref = count_refs[0]
    for other in count_refs[1:]:
        for ngram in other:
            count_ref[ngram] = max(count_ref[ngram], other[ngram])
    clipped_count = {}
    for ngram in count_hyp:
        clipped_count[ngram] = min(count_hyp[ngram], count_ref[ngram])
    matched_ngram = sum(clipped_count.values())
    total_ngram = len(hyp) - n + 1
    return matched_ngram, total_ngram


def _get_hyp_ref_length(hyp, refs):
    hyp_len = len(hyp)
    ref_lens = [len(ref) for ref in refs]
    ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - hyp_len))
    return hyp_len, ref_len


def bleu(refs_list, hypos, weights=(0.25, 0.25, 0.25, 0.25), verbose=False):
    matched_ngram = [0] * len(weights)
    total_ngram = [0] * len(weights)

    hyp_len = 0
    ref_len = 0

    for hyp, refs in zip(hypos, refs_list):
        for i in range(len(weights)):
            m, n = _count_ngram(i + 1, hyp, refs)
            matched_ngram[i] += m
            total_ngram[i] += n
        this_hyp_len, this_ref_len = _get_hyp_ref_length(hyp, refs)
        hyp_len += this_hyp_len
        ref_len += this_ref_len
    precisions = [m / n for m, n in zip(matched_ngram, total_ngram)]
    weighted_precisions = [w * (math.log(p) if p > 0 else -float('inf')) for w, p in zip(weights, precisions)]
    if hyp_len > ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - ref_len / hyp_len)
    if verbose:
        return brevity_penalty * math.exp(sum(weighted_precisions)), brevity_penalty, precisions
    return brevity_penalty * math.exp(sum(weighted_precisions))


def perplexity(cross_entropy, num_tokens, base=2) -> float:
    """Computes perplexity given the cross entropy of a sequence.

    The cross entropy is a sum over the negative log-likelihood of the sequence.
    Args:
        cross_entropy: Cross entropy of a sequence.
        num_tokens: Number of tokens in the sequence.
        base: log base.

    Returns:
        A per-token perplexity.
    """
    cross_entropy /= math.log(base) / num_tokens
    return 2 ** cross_entropy

import logging
import math
from typing import List, Callable, Any, Tuple, Dict

import torch

import thseq.data
import thseq.utils as utils

logger = logging.getLogger(__name__)


class Penalization(object):
    _SUPPORTED_PENALTY = ['gnmt', 'fairseq']

    def __init__(self, penalty: str, alpha: float, step_wise: bool) -> None:
        super().__init__()
        if penalty is not None and penalty not in Penalization._SUPPORTED_PENALTY:
            raise ValueError(f'Penalty not supported. Expected {Penalization._SUPPORTED_PENALTY}, got {penalty}.')
        self._penalty = penalty
        self._alpha = alpha
        self._step_wise = step_wise

    def enabled(self):
        return self._penalty and self._alpha != 0

    def _penalise(self, cum_scores: torch.Tensor, step: int):
        if not self.enabled():
            return cum_scores, cum_scores

        factor = 1.0
        if self._penalty == 'gnmt':
            factor = (1.0 + step / 6.0) ** (-self._alpha)
        elif self._penalty == 'fairseq':
            factor = (step + 1) ** (-self._alpha)
        return cum_scores * factor, cum_scores

    def penalise_local(self, cum_scores: torch.Tensor, step: int):
        if self._step_wise:
            return self._penalise(cum_scores, step)
        return cum_scores, cum_scores

    def penalise_global(self, cum_scores: torch.Tensor, step: int):
        return self._penalise(cum_scores, step)


def beam_search_step(k: int, cum_score: torch.Tensor, vocab_size: int):
    """
    :param k:
    :param cum_score: cumulative score of shape (B, kV)
    :param vocab_size:
    :return:
    """
    if cum_score.dim() != 2:
        raise ValueError(f'Expected input as a 2d tensor, got {cum_score.dim()}d instead.')
    top_score, top_idx = torch.topk(cum_score, min(2 * k, vocab_size), dim=-1)
    try:
        top_beam = top_idx.floor_divide(vocab_size)  # values range from 0 -> K
    except AttributeError:
        top_beam = top_idx // vocab_size
    top_idx.fmod_(vocab_size)  # values range from 0 -> V
    return top_score, top_beam, top_idx


def get_bound_length(source_length, min_len_a, min_len_b, max_len_a, max_len_b):
    min_len = source_length * min_len_a + min_len_b
    max_len = source_length * max_len_a + max_len_b
    return min_len, max_len


@torch.no_grad()
def generate(
        fn: Callable[[torch.Tensor, Any], Tuple[torch.FloatTensor, Any]],
        state: Any,
        src_lens: torch.Tensor,
        batch_size: int,
        beam_width: int,
        target_vocab: thseq.data.Vocabulary,
        penalty: str = None,
        alpha: float = None,
        step_wise_penalty: bool = None,
        min_len_a: float = 0,
        min_len_b: float = 1,
        max_len_a: float = 0,
        max_len_b: float = 200,
        block_ngram: float = 0,
        suppress_unk: bool = False,
        topk: int = 1,

) -> List[List[Dict]]:
    if block_ngram:
        logging.warning('block_ngram is not currently implemented.')
    if suppress_unk:
        logging.warning('suppress_unk is not currently implemented.')
    bsz, K = batch_size, beam_width
    bos, eos, pad, unk = target_vocab.eos_id, target_vocab.eos_id, target_vocab.pad_id, target_vocab.unk_id
    # length constraint
    min_len, max_len = get_bound_length(src_lens.max().item(), min_len_a, min_len_b, max_len_a, max_len_b)
    penalisation = Penalization(penalty, alpha, step_wise_penalty)
    # tokens and scores for alive beams
    # additional bos and eos
    tokens = src_lens.new_full((bsz * K, max_len + 2), eos).long()
    scores = src_lens.new_zeros((bsz * K, max_len + 1)).float()
    offsets = (torch.arange(bsz) * K).unsqueeze(-1).to(tokens).long()
    tokens[:, 0] = bos
    # container for generated hypotheses
    finished: List[List[Dict]] = [[] for _ in range(bsz)]

    alive2original = list(range(bsz))
    # one more step for eos
    for step in range(max_len + 1):
        inputs = tokens[:, :step + 1]  # BK x 1
        cur_bsz = inputs.size(0) // K
        if step == 0:
            inputs = tokens[offsets.view(-1), :step + 1]
            cur_bsz = inputs.size(0)
        # log_prob is of shape BK x 1 x V
        log_prob, state = fn(inputs, state)
        log_prob.squeeze_(1)  # BK x V
        # TODO: probably apply constraints individually to each sentence?
        if step < min_len:
            log_prob[:, eos] = -math.inf
        if step >= max_len:  # True when length == max_len
            log_prob[:, :eos] = -math.inf
            log_prob[:, eos + 1:] = -math.inf
        # never expand padding token
        log_prob[:, pad] = -math.inf

        # TODO: block n_gram repetition

        # accumulate scores
        cum_score = log_prob
        if step > 0:
            # BK x V
            cum_score += scores[:, step - 1].unsqueeze(-1)
        cum_score, _ = penalisation.penalise_local(cum_score, step)
        # B x 2K
        top_score, top_beam, top_idx = beam_search_step(beam_width, cum_score.view(cur_bsz, -1), log_prob.size(-1))
        top_beam = top_beam + offsets[:cur_bsz]
        is_eos = top_idx.eq(eos)
        # processes finished hypos
        # only accepts finished hypos in the top K candidates
        eos_beam = torch.masked_select(top_beam[:, :K], is_eos[:, :K])
        if eos_beam.numel() > 0:
            eos_score = torch.masked_select(top_score[:, :K], is_eos[:, :K])
            eos_score, _ = penalisation.penalise_global(eos_score, step)
            for i in range(eos_beam.numel()):
                batch_idx = eos_beam[i].item() // K
                batch_idx = alive2original[batch_idx]
                if len(finished[batch_idx]) < K:
                    finished[batch_idx].append({
                        "ids": tokens[eos_beam[i], 1:step + 2],  # including final eos
                        "score": eos_score[i]
                    })

            # finds active beams to proceed
            top_score = torch.masked_fill(top_score, is_eos, -math.inf)
            # here we need original top_beam without offset to achieve active_beam to index top_idx
            active_score, active_beam = torch.topk(top_score, K, -1)
            active_idx = torch.gather(top_idx, 1, active_beam)
            # updates active_beam after updating active_idx
            active_beam = torch.gather(top_beam, 1, active_beam)
            # ensures inactive beams will not be expanded in next step
            # active_score.masked_fill_(active_idx.eq(eos), -math.inf)

            # finds alive beams
            # a source sentence is considered alive if K candidates are not yet found
            alive_seq = [i for i in range(cur_bsz)
                         if len(finished[alive2original[i]]) < K]
            if len(alive_seq) == 0:
                break
            elif len(alive_seq) < cur_bsz:
                alive2original = [alive2original[i] for i in alive_seq]
                alive_seq = torch.tensor(alive_seq).to(tokens)
                active_score = active_score[alive_seq]
                active_beam = active_beam[alive_seq]
                active_idx = active_idx[alive_seq]
        else:
            active_score, active_beam, active_idx = top_score[:, :K], top_beam[:, :K], top_idx[:, :K]

        # yields  active_score
        active_beam = active_beam.flatten()
        # updates active beams
        tokens = tokens[active_beam]
        scores = scores[active_beam]
        tokens[:, step + 1] = active_idx.flatten()
        scores[:, step] = active_score.flatten()
        # updates state
        if step == 0:
            try:
                state = utils.select(state, active_beam.floor_divide(K))
            except AttributeError:
                state = utils.select(state, active_beam // K)
        else:
            state = utils.select(state, active_beam)

    for i, candidates in enumerate(finished):
        candidates.sort(key=lambda candidate: candidate['score'], reverse=True)
        finished[i] = candidates[:topk]
        for candidate in candidates:
            # excluding eos
            candidate['tokens'] = target_vocab.revert(candidate['ids'][:-1])
    return finished

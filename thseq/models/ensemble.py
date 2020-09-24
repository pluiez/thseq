import math
from typing import List

import torch
import torch.nn as nn

import thseq.utils as utils
from .abs import _Model, _Encoder, _Decoder


class Encoder(_Encoder):
    def __init__(self, models: List[_Model]):
        super().__init__(models[0].encoder.args, models[0].encoder.vocabulary)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return [m.encode(x) for m in self.models]


class Decoder(_Decoder):

    def __init__(self, models: List[_Model]):
        super().__init__(models[0].decoder.args, models[0].decoder.vocabulary)
        self.models = nn.ModuleList(models)

    def forward(self, y, states):
        log_probs = []
        states = []
        for i, model in enumerate(self.models):
            logit, state = model.decode(y, states[i])
            log_prob = utils.log_softmax(logit, -1)
            log_probs.append(log_prob)
            states.append(state)

        return log_probs, states


class AverageLogProb(_Model):

    def __init__(self, models: List[_Model], weights=None):
        super().__init__(models[0].args, models[0].vocabularies)
        self.encoder = Encoder(models)
        self.decoder = Decoder(models)
        self.num_model = len(models)

        self.weights = weights

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y, states):
        log_probs, states = self.decoder(y, states)
        log_probs = torch.logsumexp(torch.stack(log_probs, 0), 0) - math.log(self.num_model)
        return log_probs, states

    def initialize(self):
        pass

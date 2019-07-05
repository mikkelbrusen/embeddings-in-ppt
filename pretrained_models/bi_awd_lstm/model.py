import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            self.rnns_rev = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
                self.rnns_rev = [WeightDrop(rnn_rev, ['weight_hh_l0'], dropout=wdrop) for rnn_rev in self.rnns_rev]
        print(self.rnns)
        print(self.rnns_rev)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.rnns_rev = torch.nn.ModuleList(self.rnns_rev)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, input_rev, hidden, hidden_rev, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb_rev = embedded_dropout(self.encoder, input_rev, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)
        emb_rev = self.lockdrop(emb_rev, self.dropouti)
        raw_output = emb
        raw_output_rev = emb_rev
        new_hidden = []
        new_hidden_rev = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        raw_outputs_rev = []
        outputs = []
        outputs_rev = []
        for l, _ in enumerate(self.rnns):
            # current_input = raw_output
            raw_output, new_h = self.rnns[l](raw_output, hidden[l])
            raw_output_rev, new_h_rev = self.rnns_rev[l](raw_output_rev, hidden_rev[l])
            new_hidden.append(new_h)
            new_hidden_rev.append(new_h_rev)
            raw_outputs_rev.append(raw_output_rev)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
                raw_output_rev = self.lockdrop(raw_output_rev, self.dropouth)
                outputs_rev.append(raw_output_rev)
        hidden = new_hidden
        hidden_rev = new_hidden_rev

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        output_rev = self.lockdrop(raw_output_rev, self.dropout)
        outputs_rev.append(output_rev)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        result_rev = output_rev.view(output_rev.size(0)*output_rev.size(1), output_rev.size(2))
        if return_h:
            return result, result_rev, hidden, hidden_rev, raw_outputs, raw_outputs_rev, outputs, outputs_rev
        return result, result_rev, hidden, hidden_rev

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

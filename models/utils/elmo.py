import torch
import torch.nn as nn

from utils.utils import reverse_padded_sequence

from pretrained_models.elmo.weight_drop import WeightDrop
from pretrained_models.elmo.embed_regularize import embedded_dropout
from pretrained_models.elmo.locked_dropout import LockedDropout

class Elmo(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.1, dropouth=0.1, dropouti=0.1, dropoute=0.1, wdrop=0.1, tie_weights=False):
        super(Elmo, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
        self.rnns_rev = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]

        self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns_rev = [WeightDrop(rnn_rev, ['weight_hh_l0'], dropout=wdrop) for rnn_rev in self.rnns_rev]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.rnns_rev = torch.nn.ModuleList(self.rnns_rev)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, seq_lengths):
        input_rev = reverse_padded_sequence(input, seq_lengths, batch_first=True)
        
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0) # (bs, seq_len, emb_size)
        emb_rev = embedded_dropout(self.encoder, input_rev, dropout=self.dropoute if self.training else 0) # (bs, seq_len, emb_size)

        emb = self.lockdrop(emb, self.dropouti) #(bs, seq_len, emb_size)
        emb_rev = self.lockdrop(emb_rev, self.dropouti) #(bs, seq_len, emb_size)
        emb = emb.permute(1,0,2) # (seq_len, bs, emb_size)
        emb_rev = emb_rev.permute(1,0,2) # (seq_len, bs, emb_size)

        raw_output = emb
        raw_output_rev = emb_rev
        new_hidden = []
        new_hidden_rev = []
        raw_outputs = []
        raw_outputs_rev = []
        outputs = []
        outputs_rev = []
        for l, _ in enumerate(self.rnns):
            # current_input = raw_output
            raw_output = nn.utils.rnn.pack_padded_sequence(raw_output, seq_lengths)
            packed_output, new_h = self.rnns[l](raw_output)
            raw_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output) # (seq_len, bs, hid)

            raw_output_rev = nn.utils.rnn.pack_padded_sequence(raw_output_rev, seq_lengths)
            packed_output_rev, new_h_rev = self.rnns_rev[l](raw_output_rev)
            raw_output_rev, _ = nn.utils.rnn.pad_packed_sequence(packed_output_rev) # (seq_len, bs, hid)

            new_hidden.append(new_h)
            new_hidden_rev.append(new_h_rev)

            raw_outputs.append(raw_output)
            raw_outputs_rev.append(raw_output_rev)

            if l != self.nlayers - 1:
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
        
        return (output, output_rev), (hidden, hidden_rev), (raw_outputs, raw_outputs_rev), (outputs, outputs_rev), (emb, emb_rev)

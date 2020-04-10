import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    def __init__(self, n_class, n_layers, wordvec_size, hidden_size, dropout_p, max_length, sos_id, eos_id, device):
        super(LanguageModel, self).__init__()
        self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p, bidirectional=False).to(device)
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(n_class, self.wordvec_size)
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(self.hidden_size, n_class)
        self.device = device

    def forward_step(self, input, hidden, function=F.log_softmax):
        """ forward one time step """
        batch_size = input.size(0)
        seq_length = input.size(1)

        embedded = self.embedding(input).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, seq_length, -1)

        return predicted_softmax, hidden

    def forward(self, inputs, teacher_forcing_ratio=1.0, function=F.log_softmax):
        batch_size = inputs.size(0)
        max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        hypotheses = list()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
            predicted_softmax, hidden = self.forward_step(
                input=inputs,
                hidden=hidden,
                function=function
            )

            for di in range(predicted_softmax.size(1)):
                step_output = predicted_softmax[:, di, :]
                hypotheses.append(step_output)

        else:
            input_ = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                predicted_softmax, hidden = self.forward_step(
                    input=input_,
                    hidden=hidden,
                    function=function
                )

                step_output = predicted_softmax.squeeze(1)
                hypotheses.append(step_output)
                input_ = hypotheses[-1].topk(1)[1]

            logits = torch.stack(hypotheses, dim=1).to(self.device)
            y_hats = logits.max(-1)[1]
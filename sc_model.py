import sys
import torch
import torch.nn as nn
import torch.nn.utils

from vocab import Vocab


class SC(nn.Module):
    def __init__(self, num_layers, d_model, nhead, hidden_dim, vocab, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        super(SC, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.vocab = vocab
        self.model_embedding = nn.Embedding(len(vocab), self.d_model, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Embedding(512,self.d_model)
        self.TransformerLayer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.Transformer = nn.TransformerEncoder(self.TransformerLayer, self.num_layers, self.norm)
        self.linear = nn.Linear(self.d_model, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout)
    def forward(self, source):
        """ Take a mini-batch of source and target sentences

        @param source (List[List[str]]): list of source sentence tokens

        @returns output(tensor) shape batch, seg, rep
        """
        source_lengths = [len(s) for s in source]

        source_padded = self.vocab.to_input_tensor(source, device= self.device) # Tensor: (src_len, b)
        X = self.model_embedding(source_padded) # Tensor: (src_len, b, embedded_dim)
        mask = source_padded.permute(1,0) == self.vocab['<pad>'] # Tensor: (b, src_len)
        sequence_len = X.shape[0]
        pos = [[i for i in range(sequence_len)] for j in range(X.shape[1])] # b,src_len
        pos = torch.tensor(pos, device = self.device)
        pos_embed = self.pos_embed(pos)
        X = pos_embed.permute(1, 0, 2) + X
        context_rep = self.Transformer(X, src_key_padding_mask = mask) # Tensor: (src_len, b, embedded_dim)
        y = self.linear(context_rep) # Tensor: (src_len, b, hidden_dim)
        y = self.activation(y)
        y = self.dropout_layer(y)
        y = self.output(y) # Tensor: (src_len, b, 1)
        y = self.sigmoid(y)
        y = y.permute(1,0,2).squeeze() # Tensor: (b, src_len)
        if len(source) > 1:
            y.data.masked_fill_(mask, 0)
        return y, source_lengths

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embedding.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = SC(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(num_layers = self.num_layers,
                         d_model = self.d_model,
                         nhead = self.nhead,
                         hidden_dim = self.hidden_dim,
                         dim_feedforward= self.dim_feedforward,
                         dropout= self.dropout,
                         activation= self.activation),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

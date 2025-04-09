import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    Applies a multi-layer RNN to encode video or image features.

    Args:
        dim_vid (int): Dimension of input video/image features.
        dim_hidden (int): Dimension of the RNN hidden state.
        input_dropout_p (float, optional): Dropout probability for the input sequence (default: 0.2).
        rnn_dropout_p (float, optional): Dropout probability for the RNN layers (default: 0.5).
        n_layers (int, optional): Number of RNN layers (default: 1).
        bidirectional (bool, optional): Whether the RNN is bidirectional (default: False).
        rnn_cell (str, optional): Type of RNN cell ('gru' or 'lstm', default: 'gru').
    """

    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU
        self.rnn = self.rnn_cell(
            dim_hidden, dim_hidden, n_layers, batch_first=True,
            bidirectional=bidirectional, dropout=rnn_dropout_p if n_layers > 1 else 0
        )

        # Initialize weights
        nn.init.xavier_normal_(self.vid2hid.weight)
        nn.init.zeros_(self.vid2hid.bias)

    def forward(self, vid_feats):
        """
        Encode the input video/image features using a multi-layer RNN.

        Args:
            vid_feats (torch.Tensor): Input features, shape [batch_size, seq_len, dim_vid].

        Returns:
            tuple: (output, hidden)
                - output (torch.Tensor): Encoded features, shape [batch_size, seq_len, dim_hidden * num_directions].
                - hidden (torch.Tensor or tuple): Hidden state, shape [n_layers * num_directions, batch_size, dim_hidden].
        """
        vid_feats = vid_feats.to(self.device)
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()  # Only effective in multi-GPU settings
        output, hidden = self.rnn(vid_feats)
        return output, hidden


if __name__ == '__main__':
    # Simple test
    encoder = EncoderRNN(dim_vid=2048, dim_hidden=512)
    vid_feats = torch.randn(32, 10, 2048)  # batch_size=32, seq_len=10, dim_vid=2048
    output, hidden = encoder(vid_feats)
    print(f"Output shape: {output.shape}")  # Expected: [32, 10, 512]
    print(f"Hidden shape: {hidden.shape}")  # Expected: [1, 32, 512] for GRU
import torch
import torch.nn as nn


class S2VTModel(nn.Module):
    """
    Sequence-to-Sequence model for video/image captioning without attention.

    Args:
        vocab_size (int): Size of the vocabulary.
        max_len (int): Maximum allowed length for the sequence.
        dim_hidden (int): Dimension of the RNN hidden state.
        dim_word (int): Dimension of word embeddings.
        dim_vid (int, optional): Dimension of video/image features (default: 2048).
        sos_id (int, optional): Start of sequence token ID (default: 1).
        eos_id (int, optional): End of sequence token ID (default: 0).
        n_layers (int, optional): Number of RNN layers (default: 1).
        rnn_cell (str, optional): Type of RNN cell ('gru' or 'lstm', default: 'gru').
        rnn_dropout_p (float, optional): Dropout probability for RNN layers (default: 0.2).
    """

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, sos_id=1, eos_id=0,
                 n_layers=1, rnn_cell='gru', rnn_dropout_p=0.2):
        super(S2VTModel, self).__init__()
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU
        self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers, batch_first=True,
                                  dropout=rnn_dropout_p if n_layers > 1 else 0)
        self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers, batch_first=True,
                                  dropout=rnn_dropout_p if n_layers > 1 else 0)

        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(vocab_size, dim_word)
        self.out = nn.Linear(dim_hidden, vocab_size)

        # Initialize weights
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, vid_feats, target_variable=None, mode='train', opt=None):
        """
        Forward pass of the S2VT model.

        Args:
            vid_feats (torch.Tensor): Video/image features, shape [batch_size, n_frames, dim_vid].
            target_variable (torch.Tensor, optional): Ground truth labels, shape [batch_size, max_len].
            mode (str): 'train' or 'inference' (default: 'train').
            opt (dict, optional): Additional options (currently unused).

        Returns:
            tuple: (seq_probs, seq_preds)
                - seq_probs (torch.Tensor): Log probabilities, shape [batch_size, max_len-1, vocab_size].
                - seq_preds (torch.Tensor or list): Predicted token IDs, shape [batch_size, max_len-1] or empty list.
        """
        vid_feats = vid_feats.to(self.device)
        if target_variable is not None:
            target_variable = target_variable.to(self.device)

        batch_size, n_frames, _ = vid_feats.shape
        padding_words = torch.zeros(batch_size, n_frames, self.dim_word, device=self.device)
        padding_frames = torch.zeros(batch_size, 1, self.dim_vid, device=self.device)
        state1 = None
        state2 = None

        # Stage 1: Encode video features
        self.rnn1.flatten_parameters()  # Only effective in multi-GPU settings
        output1, state1 = self.rnn1(vid_feats, state1)
        input2 = torch.cat((output1, padding_words), dim=2)
        self.rnn2.flatten_parameters()
        output2, state2 = self.rnn2(input2, state2)

        seq_probs = []
        seq_preds = []

        if mode == 'train':
            for i in range(self.max_length - 1):
                current_words = self.embedding(target_variable[:, i])
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = torch.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, dim=1)

        else:  # Inference mode
            current_words = self.embedding(torch.full((batch_size,), self.sos_id, dtype=torch.long, device=self.device))
            for i in range(self.max_length - 1):
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = torch.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, dim=1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, dim=1)
            seq_preds = torch.cat(seq_preds, dim=1) if seq_preds else []

        return seq_probs, seq_preds


if __name__ == '__main__':
    # Simple test
    model = S2VTModel(vocab_size=10000, max_len=20, dim_hidden=512, dim_word=300)
    vid_feats = torch.randn(32, 10, 2048)  # batch_size=32, n_frames=10, dim_vid=2048
    targets = torch.randint(0, 10000, (32, 20))  # batch_size=32, max_len=20

    # Train mode
    seq_probs, seq_preds = model(vid_feats, targets, mode='train')
    print(f"Train mode - seq_probs shape: {seq_probs.shape}, seq_preds: {seq_preds}")

    # Inference mode
    seq_probs, seq_preds = model(vid_feats, mode='inference')
    print(f"Inference mode - seq_probs shape: {seq_probs.shape}, seq_preds shape: {seq_preds.shape}")
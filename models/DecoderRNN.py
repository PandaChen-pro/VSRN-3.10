import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import Attention


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): Size of the vocabulary.
        max_len (int): Maximum allowed length for the sequence to be processed.
        dim_hidden (int): Number of features in the hidden state `h`.
        dim_word (int): Dimension of word embeddings.
        n_layers (int, optional): Number of recurrent layers (default: 1).
        rnn_cell (str, optional): Type of RNN cell ('gru' or 'lstm', default: 'gru').
        bidirectional (bool, optional): Whether the encoder is bidirectional (default: False).
        input_dropout_p (float, optional): Dropout probability for the input sequence (default: 0.1).
        rnn_dropout_p (float, optional): Dropout probability for the RNN layers (default: 0.1).
    """

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, n_layers=1,
                 rnn_cell='gru', bidirectional=False, input_dropout_p=0.1, rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = 1  # Start of sequence token ID
        self.eos_id = 0  # End of sequence token ID
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(vocab_size, dim_word)
        self.attention = Attention(self.dim_hidden)
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden + dim_word, self.dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p if n_layers > 1 else 0)
        self.out = nn.Linear(self.dim_hidden, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of the output layer."""
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _init_rnn_state(self, encoder_hidden):
        """Initialize the decoder's RNN state from the encoder's hidden state."""
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):  # LSTM
            return tuple(self._cat_directions(h) for h in encoder_hidden)
        return self._cat_directions(encoder_hidden)  # GRU

    def _cat_directions(self, h):
        """Transform bidirectional hidden state into unidirectional format."""
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=2)
        return h

    def forward(self, encoder_outputs, encoder_hidden, targets=None, mode='train', opt=None):
        """
        Decode the sequence given encoder outputs and hidden state.

        Args:
            encoder_outputs (torch.Tensor): Encoder outputs, shape [batch_size, seq_len, dim_hidden * num_directions].
            encoder_hidden (torch.Tensor or tuple): Encoder hidden state.
            targets (torch.Tensor, optional): Ground truth targets, shape [batch_size, max_length].
            mode (str): 'train' or 'inference'.
            opt (dict, optional): Options including sample_max, beam_size, temperature.

        Returns:
            tuple: (seq_logprobs, seq_preds)
                - seq_logprobs (torch.Tensor): Log probabilities, shape [batch_size, max_length, vocab_size].
                - seq_preds (torch.Tensor): Predicted token IDs, shape [batch_size, max_length].
        """
        opt = opt or {}
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size = encoder_outputs.size(0)
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []

        if mode == 'train':
            targets_emb = self.embedding(targets).to(self.device)
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                logprobs = torch.log_softmax(self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))
            seq_logprobs = torch.cat(seq_logprobs, dim=1)

        elif mode == 'inference':
            if beam_size > 1:
                return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length - 1):
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                if t == 0:  # Start with <sos>
                    it = torch.full((batch_size,), self.sos_id, dtype=torch.long, device=self.device)
                elif sample_max:
                    sample_logprobs, it = torch.max(logprobs, dim=1)
                    seq_logprobs.append(sample_logprobs.unsqueeze(1))
                else:
                    prob_prev = torch.exp(logprobs / temperature if temperature != 1.0 else logprobs)
                    it = torch.multinomial(prob_prev, 1).squeeze(1)
                    sample_logprobs = logprobs.gather(1, it.unsqueeze(1))
                    seq_logprobs.append(sample_logprobs)

                seq_preds.append(it.unsqueeze(1))
                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                logprobs = torch.log_softmax(self.out(decoder_output.squeeze(1)), dim=1)

            seq_logprobs = torch.cat(seq_logprobs, dim=1)
            seq_preds = torch.cat(seq_preds, dim=1) if seq_preds else None

        return seq_logprobs, seq_preds

    def sample_beam(self, encoder_outputs, decoder_hidden, opt):
        """Beam search sampling (not implemented here)."""
        raise NotImplementedError("Beam search is not implemented in this version.")


if __name__ == '__main__':
    # Simple test
    decoder = DecoderRNN(vocab_size=10000, max_len=20, dim_hidden=512, dim_word=300)
    enc_out = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10, dim_hidden=512
    enc_hidden = torch.randn(1, 32, 512)  # num_layers=1, batch_size=32, dim_hidden=512
    targets = torch.randint(0, 10000, (32, 20))  # batch_size=32, max_len=20

    # Train mode
    seq_logprobs, _ = decoder(enc_out, enc_hidden, targets, mode='train')
    print(f"Train mode - seq_logprobs shape: {seq_logprobs.shape}")

    # Inference mode
    seq_logprobs, seq_preds = decoder(enc_out, enc_hidden, mode='inference')
    print(f"Inference mode - seq_logprobs shape: {seq_logprobs.shape}, seq_preds shape: {seq_preds.shape}")
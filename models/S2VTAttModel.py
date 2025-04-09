import torch
import torch.nn as nn


class S2VTAttModel(nn.Module):
    """
    Sequence-to-Sequence model with attention for video/image captioning.

    Args:
        encoder (nn.Module): Encoder RNN module.
        decoder (nn.Module): Decoder RNN module with attention.
    """

    def __init__(self, encoder, decoder):
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, vid_feats, target_variable=None, mode='train', opt=None):
        """
        Forward pass of the S2VT model with attention.

        Args:
            vid_feats (torch.Tensor): Video/image features, shape [batch_size, seq_len, dim_vid].
            target_variable (torch.Tensor, optional): Ground truth labels, shape [batch_size, max_len].
            mode (str): 'train' or 'inference' (default: 'train').
            opt (dict, optional): Additional options for decoding (e.g., sample_max, beam_size).

        Returns:
            tuple: (seq_prob, seq_preds)
                - seq_prob (torch.Tensor): Log probabilities, shape [batch_size, max_len-1, vocab_size].
                - seq_preds (torch.Tensor or list): Predicted token IDs, shape [batch_size, max_len-1] or empty list.
        """
        vid_feats = vid_feats.to(self.device)
        if target_variable is not None:
            target_variable = target_variable.to(self.device)

        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, opt or {})
        return seq_prob, seq_preds


if __name__ == '__main__':
    # Simple test
    from EncoderRNN import EncoderRNN
    from DecoderRNN import DecoderRNN

    encoder = EncoderRNN(dim_vid=2048, dim_hidden=512)
    decoder = DecoderRNN(vocab_size=10000, max_len=20, dim_hidden=512, dim_word=300)
    model = S2VTAttModel(encoder, decoder)
    model.to(model.device)

    vid_feats = torch.randn(32, 10, 2048)  # batch_size=32, seq_len=10, dim_vid=2048
    targets = torch.randint(0, 10000, (32, 20))  # batch_size=32, max_len=20

    # Train mode
    seq_prob, seq_preds = model(vid_feats, targets, mode='train')
    print(f"Train mode - seq_prob shape: {seq_prob.shape}, seq_preds: {seq_preds}")

    # Inference mode
    seq_prob, seq_preds = model(vid_feats, mode='inference')
    print(f"Inference mode - seq_prob shape: {seq_prob.shape}, seq_preds shape: {seq_preds.shape}")
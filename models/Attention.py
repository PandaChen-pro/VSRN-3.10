import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.

    Args:
        dim (int): Dimension of the hidden state and encoder outputs.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)

        # Initialize weights with Xavier normal
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

    def forward(self, hidden_state, encoder_outputs):
        """
        Compute the context vector using attention mechanism.

        Args:
            hidden_state (torch.Tensor): Decoder hidden state, shape [batch_size, dim]
            encoder_outputs (torch.Tensor): Encoder outputs, shape [batch_size, seq_len, dim]

        Returns:
            torch.Tensor: Context vector, shape [batch_size, dim]
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state), dim=2).view(-1, self.dim * 2)
        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context



if __name__ == '__main__':
    # Simple test
    attn = Attention(dim=512)
    hidden = torch.randn(32, 512)  # batch_size=32, dim=512
    enc_out = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10, dim=512
    context = attn(hidden, enc_out)
    print(f"Context shape: {context.shape}")  # Expected: [32, 512]
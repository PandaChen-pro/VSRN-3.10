import torch
import torch.nn as nn


def decode_sequence(ix_to_word, seq):
    """
    Decode a sequence of token indices into human-readable text.

    Args:
        ix_to_word (dict): Mapping from index (str) to word (str).
        seq (torch.Tensor): Sequence tensor, shape [N, D], where 0 is the END token.

    Returns:
        list: List of decoded strings.
    """
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        words = []
        for j in range(D):
            ix = seq[i, j].item()
            if ix == 0:  # END token
                break
            words.append(ix_to_word[str(ix)])
        out.append(' '.join(words))
    return out


class RewardCriterion(nn.Module):
    """Criterion for self-critical sequence training with rewards."""

    def __init__(self):
        super(RewardCriterion, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input, seq, reward):
        """
        Compute the loss based on input log probabilities, sequence, and rewards.

        Args:
            input (torch.Tensor): Log probabilities, shape [batch_size, seq_len, vocab_size].
            seq (torch.Tensor): Generated sequence, shape [batch_size, seq_len].
            reward (torch.Tensor): Reward values, shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        input = input.contiguous().view(-1).to(self.device)
        reward = reward.contiguous().view(-1).to(self.device)
        seq = seq.to(self.device)
        mask = (seq > 0).float()
        mask = torch.cat([torch.ones(mask.size(0), 1, device=self.device), mask[:, :-1]], dim=1).view(-1)
        output = -input * reward * mask
        return output.sum() / mask.sum()


class LanguageModelCriterion(nn.Module):
    """Criterion for language model training with negative log likelihood loss."""

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduction='none')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, logits, target, mask):
        """
        Compute the masked NLL loss for language modeling.

        Args:
            logits (torch.Tensor): Predicted log probabilities, shape [batch_size, seq_len, vocab_size].
            target (torch.Tensor): Target token indices, shape [batch_size, seq_len].
            mask (torch.Tensor): Mask for valid tokens, shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Scalar loss value averaged over batch.
        """
        logits = logits.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]

        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        loss = self.loss_fn(logits, target)
        output = (loss * mask).sum() / batch_size
        return output


if __name__ == '__main__':
    # Test decode_sequence
    ix_to_word = {'0': '<END>', '1': 'the', '2': 'cat', '3': 'sat', '4': 'on', '5': 'mat'}
    seq = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    decoded = decode_sequence(ix_to_word, seq)
    print("Decoded sequences:", decoded)

    # Test RewardCriterion
    reward_criterion = RewardCriterion()
    input_probs = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).log()  # Fake log probs
    seq = torch.tensor([[1, 2], [3, 0]])
    reward = torch.tensor([[1.0, 0.5], [0.8, 0.0]])
    loss = reward_criterion(input_probs, seq, reward)
    print(f"RewardCriterion loss: {loss.item()}")

    # Test LanguageModelCriterion
    lm_criterion = LanguageModelCriterion()
    logits = torch.randn(2, 3, 5)  # batch_size=2, seq_len=3, vocab_size=5
    target = torch.tensor([[1, 2, 0], [3, 4, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float)
    loss = lm_criterion(logits, target, mask)
    print(f"LanguageModelCriterion loss: {loss.item()}")
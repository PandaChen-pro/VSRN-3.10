import numpy as np
from collections import OrderedDict
import torch

try:
    from pycocoevalcap.cider.ciderD import CiderD
except ImportError:
    raise ImportError("Please install pycocoevalcap: pip install pycocoevalcap")


def init_cider_scorer(cached_tokens):
    """
    Initialize the CiderD scorer with cached tokens.

    Args:
        cached_tokens (str): Path to cached tokens file or corpus identifier.

    Returns:
        CiderD: Initialized CiderD scorer instance.
    """
    return CiderD(df=cached_tokens)


def array_to_str(arr):
    """
    Convert an array of token IDs to a string, stopping at 0 (EOS).

    Args:
        arr (numpy.ndarray): Array of token IDs.

    Returns:
        str: Space-separated string of token IDs.
    """
    return ' '.join(str(int(x)) for x in arr if x != 0)


def get_self_critical_reward(model, fc_feats, data, gen_result, cached_tokens='msr-all-idxs'):
    """
    Compute self-critical reward using CiderD scores.

    Args:
        model (nn.Module): Captioning model.
        fc_feats (torch.Tensor): Input features, shape [batch_size, seq_len, dim_vid].
        data (dict): Ground truth data containing 'gts' tensor.
        gen_result (torch.Tensor): Generated captions, shape [batch_size, max_len].
        cached_tokens (str, optional): Cached tokens identifier (default: 'msr-all-idxs').

    Returns:
        numpy.ndarray: Rewards array, shape [batch_size, max_len].
    """
    device = next(model.parameters()).device
    fc_feats = fc_feats.to(device)
    gen_result = gen_result.to(device)

    batch_size = gen_result.size(0)

    # Get greedy decoding baseline
    with torch.no_grad():
        _, greedy_res = model(fc_feats, mode='inference')

    # Prepare results and ground truth
    res = OrderedDict()
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    gt_tensor = data['gts'].cpu().numpy()
    for i in range(gt_tensor.shape[0]):
        gts[i] = [array_to_str(gt_tensor[i, j]) for j in range(gt_tensor.shape[1]) if gt_tensor[i, j].sum() > 0]

    # Format for CiderD
    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    # Compute CiderD scores
    cider_scorer = init_cider_scorer(cached_tokens)
    cider_score, scores = cider_scorer.compute_score(gts, res)
    print('Cider scores:', cider_score)

    # Compute rewards
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], axis=1)

    return rewards


if __name__ == '__main__':
    # Simple test
    class DummyModel(torch.nn.Module):
        def forward(self, fc_feats, mode='train'):
            batch_size = fc_feats.size(0)
            if mode == 'inference':
                return None, torch.randint(0, 100, (batch_size, 19))
            return torch.randn(batch_size, 19, 100), None

    model = DummyModel()
    fc_feats = torch.randn(2, 10, 2048)  # batch_size=2, seq_len=10, dim_vid=2048
    data = {'gts': torch.randint(0, 100, (2, 5, 20))}  # 2 samples, 5 GTs, max_len=20
    gen_result = torch.randint(0, 100, (2, 19))  # batch_size=2, max_len-1=19

    rewards = get_self_critical_reward(model, fc_feats, data, gen_result)
    print(f"Rewards shape: {rewards.shape}, Rewards: {rewards}")
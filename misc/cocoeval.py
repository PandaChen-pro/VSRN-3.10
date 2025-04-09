'''
Wrapper for evaluation on CIDEr, ROUGE_L, METEOR, and Bleu_N using pycocoevalcap.
Adapted from https://github.com/yaoli/arctic-capgen-vid and https://github.com/tylin/coco-caption.
'''

import json
import os
from contextlib import redirect_stdout, redirect_stderr

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
except ImportError:
    raise ImportError("Please install pycocoevalcap: pip install pycocoevalcap")


class COCOScorer:
    """COCO evaluation scorer for captioning metrics."""

    def __init__(self):
        print('Initializing COCO-EVAL scorer')

    def score(self, GT, RES, IDs):
        """
        Compute evaluation scores for given ground truth and generated captions.

        Args:
            GT (dict): Ground truth captions, {ID: [{image_id, caption}, ...]}.
            RES (dict): Generated captions, {ID: [{image_id, caption}, ...]}.
            IDs (list): List of image IDs to evaluate.

        Returns:
            dict: Evaluation scores for each metric.
        """
        self.eval = {}
        self.imgToEval = {}
        gts = {ID: GT[ID] for ID in IDs}
        res = {ID: RES[ID] for ID in IDs}

        print('Tokenizing...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        print('Setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        print('Computing scores...')
        for scorer, method in scorers:
            print(f'Computing {scorer.method()} score...')
            with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
                score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print(f"{m}: {sc:.3f}")
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print(f"{method}: {score:.3f}")

        return self.eval

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {"image_id": imgId}
            self.imgToEval[imgId][method] = score


def score(ref, sample):
    """
    Compute evaluation scores for reference and sample captions.

    Args:
        ref (dict): Reference captions, {ID: [{image_id, caption}, ...]}.
        sample (dict): Sample captions, {ID: [{image_id, caption}, ...]}.

    Returns:
        dict: Final scores for each metric.
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        print(f'Computing {scorer.method()} score with COCO-EVAL...')
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            score, _ = scorer.compute_score(ref, sample)
        if isinstance(method, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def test_cocoscorer():
    """Test the COCOScorer with sample data."""
    gts = {
        '184321': [
            {'image_id': '184321', 'cap_id': 0, 'caption': 'A train traveling down tracks next to lights.',
             'tokenized': 'a train traveling down tracks next to lights'},
            {'image_id': '184321', 'cap_id': 1, 'caption': 'A train coming down the tracks arriving at a station.',
             'tokenized': 'a train coming down the tracks arriving at a station'}],
        '81922': [
            {'image_id': '81922', 'cap_id': 0, 'caption': 'A large jetliner flying over a traffic filled street.',
             'tokenized': 'a large jetliner flying over a traffic filled street'},
            {'image_id': '81922', 'cap_id': 1, 'caption': 'The plane is flying over top of the cars',
             'tokenized': 'the plane is flying over top of the cars'}]
    }

    samples = {
        '184321': [{'image_id': '184321', 'caption': 'train traveling down a track in front of a road'}],
        '81922': [{'image_id': '81922', 'caption': 'plane is flying through the sky'}],
    }
    IDs = ['184321', '81922']

    scorer = COCOScorer()
    scores = scorer.score(gts, samples, IDs)
    print("Evaluation scores:", scores)


if __name__ == '__main__':
    test_cocoscorer()
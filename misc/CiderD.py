# 在/home/code/VSRN/misc/ciderD/ciderD.py文件中添加下面的代码

#!/usr/bin/env python
# 
# File Name : ciderD.py
#
# Description : Describes the class to compute the CIDEr-D (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date : 2015-06-24
# Last Modified : 2015-06-24
# Authors : Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from ciderD_scorer import CiderScorer
import pdb

class CiderD:
    """
    Main Class to compute the CIDEr metric 
    """
    def __init__(self, n=4, sigma=6.0, df="corpus"):
        """
        Initialize the CIDEr-D scorer
        n: n-gram size
        sigma: sigma for gaussian penalty
        df: document-frequency definition to use (default: corpus)
            - "corpus": use document frequencies from the corpus of references
            - "coco-val-df": use pre-computed document frequencies from the coco dataset
        """
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        # set the df to use the corpus
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param gts: dict of dict containing image id's and references as keys
        :param res: dict containing image id's and candidate as keys
        :return: Overall CIDEr score, and CIDEr score for each image
        """
        # clear all the previous hypos and refs
        self.cider_scorer.clear()
        
        for id in gts.keys():
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(isinstance(hypo, list))
            assert(isinstance(ref, list))
            assert(len(hypo) == 1)
            assert(len(ref) >= 1)

            self.cider_scorer.cook_append(hypo, ref)

        (score, scores) = self.cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr-D"

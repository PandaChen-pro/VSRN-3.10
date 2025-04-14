import os
import pickle
import time
import numpy as np
import torch
from data import get_test_loader
from vocab import Vocabulary  # NOQA
from model import VSRN, order_sim
from collections import OrderedDict


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-4)

    def __str__(self):
        if self.count == 0:
            return str(self.val)
        return f'{self.val:.4f} ({self.avg:.4f})'


class LogCollector:
    """A collection of logging objects that can change from train to val."""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        return '  '.join(f'{k} {str(v)}' for k, v in self.meters.items())

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, step)


def encode_data(model, data_loader):
    """Encode all images and captions from the data_loader."""
    batch_time = AverageMeter()
    val_logger = LogCollector()
    model.val_start()

    device = next(model.parameters()).device
    img_embs = None
    cap_embs = None

    with torch.no_grad():
        for i, (images, captions, lengths, ids, caption_labels, caption_masks) in enumerate(data_loader):
            start = time.time()
            images = images.to(device)
            captions = captions.to(device)

            model.logger = val_logger
            img_emb, cap_emb, _ = model.forward_emb(images, captions, lengths)

            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

            img_embs[ids] = img_emb.cpu().numpy()
            cap_embs[ids] = cap_emb.cpu().numpy()

            batch_time.update(time.time() - start)
            if i % 10 == 0:
                print(f'Encoding: [{i}/{len(data_loader)}] Time {batch_time}')

    return img_embs, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on dev or test split.

    Args:
        model_path (str): Path to the trained model checkpoint.
        data_path (str, optional): Override data path in checkpoint options.
        split (str): Dataset split to evaluate ('dev' or 'test', default: 'dev').
        fold5 (bool): If True, perform 5-fold cross-validation (MSCOCO only).

    Returns:
        None (prints evaluation metrics and saves ranks).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    with open(os.path.join(opt.vocab_path, f'{opt.data_name}_vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    model = VSRN(opt).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print(f'Images: {img_embs.shape[0] // 5}, Captions: {cap_embs.shape[0]}')

    if not fold5:
        r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print(f"rsum: {rsum:.1f}")
        print(f"Average i2t Recall: {ar:.1f}")
        print(f"Image to text: {r[0]:.1f} {r[1]:.1f} {r[2]:.1f} {r[3]:.1f} {r[4]:.1f}")
        print(f"Average t2i Recall: {ari:.1f}")
        print(f"Text to image: {ri[0]:.1f} {ri[1]:.1f} {ri[2]:.1f} {ri[3]:.1f} {ri[4]:.1f}")
    else:
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                         measure=opt.measure, return_ranks=True)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                           measure=opt.measure, return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print(f"Fold {i}:")
            print(f"Image to text: {r[0]:.1f} {r[1]:.1f} {r[2]:.1f} {r[3]:.1f} {r[4]:.1f}")
            print(f"Text to image: {ri[0]:.1f} {ri[1]:.1f} {ri[2]:.1f} {ri[3]:.1f} {ri[4]:.1f}")
            print(f"rsum: {rsum:.1f} ar: {ar:.1f} ari: {ari:.1f}")
            results.append(list(r) + list(ri) + [ar, ari, rsum])

        mean_metrics = tuple(np.array(results).mean(axis=0))
        print("-----------------------------------")
        print("Mean metrics:")
        print(f"rsum: {mean_metrics[10] * 6:.1f}")
        print(f"Average i2t Recall: {mean_metrics[11]:.1f}")
        print(f"Image to text: {mean_metrics[0]:.1f} {mean_metrics[1]:.1f} {mean_metrics[2]:.1f} "
              f"{mean_metrics[3]:.1f} {mean_metrics[4]:.1f}")
        print(f"Average t2i Recall: {mean_metrics[12]:.1f}")
        print(f"Text to image: {mean_metrics[5]:.1f} {mean_metrics[6]:.1f} {mean_metrics[7]:.1f} "
              f"{mean_metrics[8]:.1f} {mean_metrics[9]:.1f}")

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Image-to-Text evaluation (Image Annotation).

    Args:
        images (numpy.ndarray): Image embeddings, shape [5N, K].
        captions (numpy.ndarray): Caption embeddings, shape [5N, K].
        npts (int, optional): Number of points (default: images.shape[0] / 5).
        measure (str): Similarity measure ('cosine' or 'order').
        return_ranks (bool): If True, return ranks and top1 indices.

    Returns:
        tuple: (r1, r5, r10, medr, meanr) or ((r1, r5, r10, medr, meanr), (ranks, top1)).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    npts = npts or images.shape[0] // 5
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        im = images[5 * index].reshape(1, -1)
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d = order_sim(torch.from_numpy(im2).to(device), torch.from_numpy(captions).to(device))
                d = d.cpu().numpy()
            d = d[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        rank = min(np.where(inds == i)[0][0] for i in range(5 * index, 5 * index + 5))
        ranks[index] = rank
        top1[index] = inds[0]

    r1 = 100.0 * np.sum(ranks < 1) / len(ranks)
    r5 = 100.0 * np.sum(ranks < 5) / len(ranks)
    r10 = 100.0 * np.sum(ranks < 10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10, medr, meanr), (ranks, top1) if return_ranks else (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text-to-Image evaluation (Image Search).

    Args:
        images (numpy.ndarray): Image embeddings, shape [5N, K].
        captions (numpy.ndarray): Caption embeddings, shape [5N, K].
        npts (int, optional): Number of points (default: images.shape[0] / 5).
        measure (str): Similarity measure ('cosine' or 'order').
        return_ranks (bool): If True, return ranks and top1 indices.

    Returns:
        tuple: (r1, r5, r10, medr, meanr) or ((r1, r5, r10, medr, meanr), (ranks, top1)).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    npts = npts or images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    for index in range(npts):
        queries = captions[5 * index:5 * index + 5]
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d = order_sim(torch.from_numpy(ims).to(device), torch.from_numpy(q2).to(device))
                d = d.cpu().numpy()
            d = d[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.argsort(d, axis=1)[:, ::-1]
        for i in range(len(inds)):
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    r1 = 100.0 * np.sum(ranks < 1) / len(ranks)
    r5 = 100.0 * np.sum(ranks < 5) / len(ranks)
    r10 = 100.0 * np.sum(ranks < 10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10, medr, meanr), (ranks, top1) if return_ranks else (r1, r5, r10, medr, meanr)


if __name__ == '__main__':
    # Mock test (requires actual model and data)
    opt = type('Opt', (), {
        'data_name': 'coco', 'vocab_path': './vocab', 'crop_size': 224,
        'batch_size': 32, 'workers': 0, 'measure': 'cosine'
    })()
    evalrank('path_to_model.pth.tar', data_path='path_to_data', split='test', fold5=False)
# -*- coding: utf-8 -*-
# ^ Added encoding declaration for potentially non-ASCII characters in comments/strings

import os
import pickle
import time
from collections import OrderedDict

import numpy
import numpy as np # Use consistent alias
import torch
# Removed: from __future__ import print_function (not needed in Python 3)

# Assuming these imports are correct relative to your project structure
from data import get_test_loader
from vocab import Vocabulary  # NOQA
from model import VSRN, order_sim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # Default n=1 if not provided makes more sense
        self.val = val
        self.sum += val * n
        self.count += n
        # Ensure float division and avoid division by zero
        self.avg = self.sum / (self.count + 1e-6) # Slightly more robust than .0001

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        # %.4f formatting works in Python 3 too
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=1): # Default n=1
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        # Changed: iteritems() -> items()
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        # Changed: iteritems() -> items()
        for k, v in self.meters.items():
            # Assuming tb_logger has a method like log_value or add_scalar
            # This part might need adjustment based on your tensorboard logger library
            # Using add_scalar which is common (e.g., tensorboardX or torch.utils.tensorboard)
            if hasattr(tb_logger, 'add_scalar'):
                 tb_logger.add_scalar(prefix + k, v.val, global_step=step)
            elif hasattr(tb_logger, 'log_value'): # Keep original if needed
                 tb_logger.log_value(prefix + k, v.val, step=step)
            # else: print warning or raise error


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval() # Use standard PyTorch method name
    if hasattr(model, 'val_start'): # Keep custom method if it does more
        model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    fc_img_emds = None # Assuming this might be used later or was part of VSRN

    # Changed: Use torch.no_grad() context manager for inference
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            # Unpack batch data - ensure this matches your DataLoader output
            # Assuming standard (images, captions, lengths, ids, ...) structure
            # Adapt this unpacking if your data loader returns something different
            images, captions, lengths, ids = batch_data[:4]
            # Handle optional extra data like labels/masks if present
            caption_labels = batch_data[4] if len(batch_data) > 4 else None
            caption_masks = batch_data[5] if len(batch_data) > 5 else None

            # make sure val logger is used (if model uses it internally)
            if hasattr(model, 'logger'):
                model.logger = val_logger

            # Ensure tensors are on the correct device (e.g., GPU if available)
            # This assumes model parameters are already on the correct device
            if torch.cuda.is_available():
                 images = images.cuda()
                 captions = captions.cuda()
                 # lengths might not need .cuda() depending on usage
                 # caption_labels, caption_masks if they exist and are tensors

            # compute the embeddings
            # Removed volatile=True
            img_emb, cap_emb, fc_img_emd = model.forward_emb(images, captions, lengths)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)), dtype=np.float32)
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)), dtype=np.float32)
                if fc_img_emd is not None: # Handle optional fc_img_emd
                     fc_img_emds = np.zeros((len(data_loader.dataset), fc_img_emd.size(1), fc_img_emd.size(2)), dtype=np.float32)


            # preserve the embeddings by copying from gpu and converting to numpy
            # Changed: .data -> .detach()
            img_embs[ids] = img_emb.detach().cpu().numpy().copy()
            cap_embs[ids] = cap_emb.detach().cpu().numpy().copy()
            if fc_img_emd is not None and fc_img_emds is not None:
                 fc_img_emds[ids] = fc_img_emd.detach().cpu().numpy().copy()


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging(f'Encoding: [{i}/{len(data_loader)}] time: {batch_time}') # Use f-string

            # Explicitly delete tensors to potentially free GPU memory sooner
            del images, captions, img_emb, cap_emb, fc_img_emd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    # Depending on whether fc_img_emds was used/needed, you might return it too
    # return img_embs, cap_embs, fc_img_emds
    return img_embs, cap_embs


def evalrank(model_path, model_path2, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    # Ensure map_location is used if loading GPU model on CPU or vice-versa
    if torch.cuda.is_available():
        map_location = None # Loads to GPU if saved on GPU, CPU if saved on CPU
    else:
        map_location = torch.device('cpu') # Loads to CPU

    checkpoint = torch.load(model_path, map_location=map_location)
    opt = checkpoint['opt']

    checkpoint2 = torch.load(model_path2, map_location=map_location)
    opt2 = checkpoint2['opt']


    # Update data path if provided
    if data_path is not None:
        opt.data_path = data_path
        # Assuming both models use the same data path structure
        opt2.data_path = data_path


    # load vocabulary used by the model
    vocab_path = os.path.join(opt.vocab_path, f'{opt.data_name}_vocab.pkl')
    print(f"Loading vocabulary from: {vocab_path}") # Use f-string
    with open(vocab_path, 'rb') as f:
        # Changed: Added encoding='latin1' for Python 2 compatibility
        vocab = pickle.load(f, encoding='latin1')
    opt.vocab_size = len(vocab)
    opt2.vocab_size = len(vocab) # Assuming same vocab for both models


    # construct model
    model = VSRN(opt)
    model2 = VSRN(opt2)

    # load model state
    model.load_state_dict(checkpoint['model'])
    model2.load_state_dict(checkpoint2['model'])

    # Move models to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    model2 = model2.to(device)

    print('Loading dataset') # Changed: print()
    # Pass opt (or opt2, assuming relevant params like batch_size are same)
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...') # Changed: print()
    # Pass models already moved to the correct device
    img_embs, cap_embs = encode_data(model, data_loader)
    img_embs2, cap_embs2 = encode_data(model2, data_loader)

    # Changed: // for integer division, use f-strings
    # Ensure cap_embs has the same first dimension size as img_embs
    print(f'Images: {img_embs.shape[0] // 5}, Captions: {cap_embs.shape[0]} (should be {img_embs.shape[0]})')
    if img_embs.shape[0] != cap_embs.shape[0]:
         print(f"Warning: Mismatch in number of image ({img_embs.shape[0]}) and caption ({cap_embs.shape[0]}) embeddings!")
         # Decide how to handle mismatch, e.g. use minimum, error out, etc.
         # Assuming they should match for standard evaluation:
         # min_len = min(img_embs.shape[0], cap_embs.shape[0])
         # img_embs = img_embs[:min_len]
         # cap_embs = cap_embs[:min_len]
         # print(f"Using minimum length: {min_len}")


    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, img_embs2, cap_embs2, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, img_embs2, cap_embs2,
                      measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        # Changed: Use f-strings for printing
        print(f"rsum: {rsum:.1f}")
        print(f"Average i2t Recall: {ar:.1f}")
        print(f"Image to text: {r[0]:.1f} {r[1]:.1f} {r[2]:.1f} {r[3]:.1f} {r[4]:.1f}")
        print(f"Average t2i Recall: {ari:.1f}")
        print(f"Text to image: {ri[0]:.1f} {ri[1]:.1f} {ri[2]:.1f} {ri[3]:.1f} {ri[4]:.1f}")
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        # Ensure fold size is integer
        fold_size = 5000
        n_folds = 5
        # Changed: Use f-strings for printing
        print(f"Using {n_folds}-fold validation with fold size {fold_size}")

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            print(f"--- Fold {i+1}/{n_folds} ---")

            # Slice embeddings for the current fold
            img_embs_fold = img_embs[start_idx:end_idx]
            cap_embs_fold = cap_embs[start_idx:end_idx]
            img_embs2_fold = img_embs2[start_idx:end_idx]
            cap_embs2_fold = cap_embs2[start_idx:end_idx]

            r, rt0 = i2t(img_embs_fold, cap_embs_fold,
                         img_embs2_fold, cap_embs2_fold,
                         measure=opt.measure, return_ranks=True)
            print(f"Image to text: {r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}, {r[3]:.1f}, {r[4]:.1f}")

            ri, rti0 = t2i(img_embs_fold, cap_embs_fold,
                           img_embs2_fold, cap_embs2_fold,
                           measure=opt.measure, return_ranks=True)
            print(f"Text to image: {ri[0]:.1f}, {ri[1]:.1f}, {ri[2]:.1f}, {ri[3]:.1f}, {ri[4]:.1f}")

            # Store ranks from the first fold (assuming this is desired behavior)
            if i == 0:
                rt, rti = rt0, rti0

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print(f"rsum: {rsum:.1f} ar: {ar:.1f} ari: {ari:.1f}")
            results.append(list(r) + list(ri) + [ar, ari, rsum])

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        # Assuming mean_metrics indices match the order in 'results'
        # [r1, r5, r10, medr, meanr, ri1, ri5, ri10, rimedr, rimeanr, ar, ari, rsum]
        print(f"rsum: {mean_metrics[12]:.1f}") # Index 12 is rsum
        print(f"Average i2t Recall: {mean_metrics[10]:.1f}") # Index 10 is ar
        print(f"Image to text: {mean_metrics[0]:.1f} {mean_metrics[1]:.1f} {mean_metrics[2]:.1f} {mean_metrics[3]:.1f} {mean_metrics[4]:.1f}")
        print(f"Average t2i Recall: {mean_metrics[11]:.1f}") # Index 11 is ari
        print(f"Text to image: {mean_metrics[5]:.1f} {mean_metrics[6]:.1f} {mean_metrics[7]:.1f} {mean_metrics[8]:.1f} {mean_metrics[9]:.1f}")

    # Consider saving ranks with a more descriptive name if needed
    rank_save_path = 'ranks.pth.tar'
    print(f"Saving ranks to {rank_save_path}")
    torch.save({'rt': rt, 'rti': rti}, rank_save_path)


def i2t(images, captions, images2, captions2, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, K) matrix of images (N = 5 * num_images)
    Captions: (N, K) matrix of captions
    """
    n_total_items = images.shape[0]
    captions_per_image = 5 # Assuming 5 captions per image based on original code

    if npts is None:
        # Changed: // for integer division
        npts = n_total_items // captions_per_image
    else:
        # Ensure npts doesn't exceed available data
        npts = min(npts, n_total_items // captions_per_image)


    # Validate shapes if possible
    if images.shape[0] != captions.shape[0] or images2.shape[0] != captions2.shape[0] or images.shape[0] != images2.shape[0]:
         raise ValueError("Shape mismatch between image/caption embeddings")
    if images.shape[1] != images2.shape[1] or captions.shape[1] != captions2.shape[1]:
         raise ValueError("Dimension mismatch between model1 and model2 embeddings")


    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    index_list = [] # This was computed but not used, keep if needed elsewhere

    # Check if CUDA is available and embeddings are large enough for GPU use
    use_cuda = torch.cuda.is_available() and captions.nbytes > 1e7 # Heuristic: use GPU for large data

    if use_cuda:
        # Move caption embeddings to GPU once
        # Changed: Use torch.from_numpy
        captions_gpu = torch.from_numpy(captions).cuda()
        captions2_gpu = torch.from_numpy(captions2).cuda()

    for index in range(npts):
        # Get query image
        # Note: Original code slices images[5*index], implying only first image of 5 is used as query
        query_idx = index * captions_per_image
        im = images[query_idx].reshape(1, images.shape[1])
        im_2 = images2[query_idx].reshape(1, images2.shape[1])

        # Compute scores
        if measure == 'order':
            # Convert query image to tensor and move to GPU if needed
            # Changed: Use torch.from_numpy
            im_tensor = torch.from_numpy(im)
            im_2_tensor = torch.from_numpy(im_2)
            if use_cuda:
                im_tensor = im_tensor.cuda()
                im_2_tensor = im_2_tensor.cuda()

            # order_sim expects batches, calculate for this single image query
            # Ensure order_sim handles (1, K) query vs (N, K) captions
            d = order_sim(im_tensor, captions_gpu if use_cuda else torch.from_numpy(captions))
            d2 = order_sim(im_2_tensor, captions2_gpu if use_cuda else torch.from_numpy(captions2))

            # Move results back to CPU if necessary and flatten
            d = d.cpu().numpy().flatten()
            d2 = d2.cpu().numpy().flatten()

        else: # Default 'cosine' similarity assumed via dot product
            if use_cuda:
                # Compute dot product on GPU
                # Changed: Use torch.from_numpy
                im_tensor = torch.from_numpy(im).cuda()
                im_2_tensor = torch.from_numpy(im_2).cuda()
                # Use matmul for clarity, ensure shapes are (1, K) @ (K, N) -> (1, N)
                d = torch.matmul(im_tensor, captions_gpu.t()).cpu().numpy().flatten()
                d2 = torch.matmul(im_2_tensor, captions2_gpu.t()).cpu().numpy().flatten()
            else:
                # Compute dot product on CPU
                d = np.dot(im, captions.T).flatten()
                d2 = np.dot(im_2, captions2.T).flatten()

        # Average scores from both models
        d_avg = (d + d2) / 2.0

        inds = np.argsort(d_avg)[::-1]
        index_list.append(inds[0]) # Keep track of top-1 index if needed

        # Score: Find rank of the *ground truth* captions for this image
        rank = float('inf') # Use float('inf') for comparison
        # Ground truth captions are indices [query_idx, query_idx + captions_per_image)
        for i in range(query_idx, query_idx + captions_per_image):
            # Find the position (rank) of caption 'i' in the sorted list 'inds'
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0] # Store the index of the top-ranked caption

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # Changed: Cast median/mean rank to int
    medr = int(np.floor(np.median(ranks)) + 1)
    meanr = int(np.floor(ranks.mean()) + 1) # Usually mean rank isn't floored, but follow original if needed

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, images2, captions2, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    n_total_items = images.shape[0]
    captions_per_image = 5 # Assuming 5 captions per image

    if npts is None:
        # Changed: // for integer division
        npts = n_total_items // captions_per_image
    else:
        npts = min(npts, n_total_items // captions_per_image)


    # Validate shapes
    if images.shape[0] != captions.shape[0] or images2.shape[0] != captions2.shape[0] or images.shape[0] != images2.shape[0]:
         raise ValueError("Shape mismatch between image/caption embeddings")
    if images.shape[1] != images2.shape[1] or captions.shape[1] != captions2.shape[1]:
         raise ValueError("Dimension mismatch between model1 and model2 embeddings")


    # Get unique image embeddings (assuming images[0], images[5], images[10]...)
    # Use array slicing for efficiency
    unique_image_indices = np.arange(0, n_total_items, captions_per_image)
    ims = images[unique_image_indices]
    ims2 = images2[unique_image_indices]

    n_images = ims.shape[0] # Should be equal to npts

    ranks = np.zeros(n_total_items) # Rank for each caption query
    top1 = np.zeros(n_total_items) # Top-1 image index for each caption query


    # Check if CUDA is available and embeddings are large enough for GPU use
    use_cuda = torch.cuda.is_available() and ims.nbytes > 1e7 # Heuristic

    if use_cuda:
        # Move unique image embeddings to GPU once
        # Changed: Use torch.from_numpy
        ims_gpu = torch.from_numpy(ims).cuda()
        ims2_gpu = torch.from_numpy(ims2).cuda()

    # Process captions in chunks to potentially manage memory if N is very large
    # Using a chunk size suitable for typical memory, adjust if needed
    chunk_size = 128
    for i_chunk in range(0, n_total_items, chunk_size):
         chunk_end = min(i_chunk + chunk_size, n_total_items)
         # Get query captions for the current chunk
         queries = captions[i_chunk:chunk_end]
         queries2 = captions2[i_chunk:chunk_end]

         # Compute scores for the chunk
         if measure == 'order':
             # Convert query captions to tensor and move to GPU if needed
             # Changed: Use torch.from_numpy
             q_tensor = torch.from_numpy(queries)
             q2_tensor = torch.from_numpy(queries2)
             if use_cuda:
                 q_tensor = q_tensor.cuda()
                 q2_tensor = q2_tensor.cuda()

             # order_sim expects (batch, K) for both args.
             # Query: (chunk_size, K), Images: (n_images, K)
             # Need to ensure order_sim supports this or transpose image tensor if needed.
             # Assuming order_sim(images, captions) returns (n_images, chunk_size)
             d = order_sim(ims_gpu if use_cuda else torch.from_numpy(ims), q_tensor)
             d2 = order_sim(ims2_gpu if use_cuda else torch.from_numpy(ims2), q2_tensor)

             # Move results back to CPU if necessary and transpose to (chunk_size, n_images)
             d = d.cpu().numpy().T
             d2 = d2.cpu().numpy().T

         else: # Default 'cosine' similarity assumed via dot product
             if use_cuda:
                 # Compute dot product on GPU
                 # Changed: Use torch.from_numpy
                 q_tensor = torch.from_numpy(queries).cuda()
                 q2_tensor = torch.from_numpy(queries2).cuda()
                 # Use matmul: (chunk_size, K) @ (K, n_images) -> (chunk_size, n_images)
                 d = torch.matmul(q_tensor, ims_gpu.t()).cpu().numpy()
                 d2 = torch.matmul(q2_tensor, ims2_gpu.t()).cpu().numpy()
             else:
                 # Compute dot product on CPU
                 d = np.dot(queries, ims.T)
                 d2 = np.dot(queries2, ims2.T)

         # Average scores from both models
         d_avg = (d + d2) / 2.0 # Shape: (chunk_size, n_images)

         # Process results for each query in the chunk
         for i_local, scores in enumerate(d_avg):
             global_idx = i_chunk + i_local # Index in the original captions array
             inds = np.argsort(scores)[::-1] # Sorted image indices for this caption query

             # Find the rank of the *correct* image
             # Correct image index is global_idx // captions_per_image
             correct_image_idx = global_idx // captions_per_image
             rank = np.where(inds == correct_image_idx)[0][0]

             ranks[global_idx] = rank
             top1[global_idx] = inds[0] # Store the index of the top-ranked image


    # Compute metrics over all caption queries
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # Changed: Cast median/mean rank to int
    medr = int(np.floor(np.median(ranks)) + 1)
    meanr = int(np.floor(ranks.mean()) + 1) # Again, floor might not be standard for meanr

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


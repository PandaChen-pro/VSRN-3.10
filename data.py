import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    """
    roots = {}
    ids = {}
    if name == 'coco':
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy'))
        )
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif name == 'f8k':
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif name == 'f30k':
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None, ids=None):
        self.root = root
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        if ids is None:
            self.ids = list(self.coco[0].anns.keys())
        else:
            self.ids = ids

        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        vocab = self.vocab
        root, caption, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [vocab('<start>')] + [vocab(token) for token in tokens] + [vocab('<end>')]
        target = torch.tensor(caption)
        return image, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        return root, caption, img_id, path, image

    def __len__(self):
        return len(self.ids)


class FlickrDataset(data.Dataset):
    """Dataset loader for Flickr30k and Flickr8k full datasets."""

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = [(i, x) for i, d in enumerate(self.dataset)
                    if d['split'] == split for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        vocab = self.vocab
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [vocab('<start>')] + [vocab(token) for token in tokens] + [vocab('<end>')]
        target = torch.tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)


class PrecompDataset(data.Dataset):
    """Load precomputed captions and image features."""

    def __init__(self, data_path, data_split, vocab, opt):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        token_caption = []
        with open(loc + f'{data_split}_caps.txt', 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())
                tokens = nltk.tokenize.word_tokenize(line.strip().lower())
                token_caption.append(tokens)

        each_cap_lengths = [len(cap) for cap in token_caption]
        calculate_max_len = max(each_cap_lengths) + 2
        print(calculate_max_len)

        # Image features
        self.images = np.load(loc + f'{data_split}_ims.npy')
        self.length = len(self.captions)
        self.im_div = 5 if self.images.shape[0] != self.length else 1
        if data_split == 'dev':
            self.length = 5000

        self.max_len = opt.max_len

    def __getitem__(self, index):
        img_id = index // self.im_div
        image = torch.tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = [vocab('<start>')] + [vocab(token) for token in tokens] + [vocab('<end>')]
        target = torch.tensor(caption)

        # Deal with caption model data
        mask = np.zeros(self.max_len + 1)
        gts = np.zeros(self.max_len + 1)
        cap_caption = ['<start>'] + tokens + ['<end>']
        if len(cap_caption) > self.max_len - 1:
            cap_caption = cap_caption[:self.max_len]
            cap_caption[-1] = '<end>']

        for j, w in enumerate(cap_caption):
            gts[j] = vocab(w)

        non_zero = np.nonzero(gts == 0)[0]
        mask[:int(non_zero[0]) + 1] = 1 if len(non_zero) > 0 else 0

        caption_label = torch.tensor(gts, dtype=torch.long)
        caption_mask = torch.tensor(mask, dtype=torch.float)
        return image, target, index, img_id, caption_label, caption_mask

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples."""
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, caption_labels, caption_masks = zip(*data)

    images = torch.stack(images, 0)
    caption_labels_ = torch.stack(caption_labels, 0)
    caption_masks_ = torch.stack(caption_masks, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids, caption_labels_, caption_masks_


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True, num_workers=2, ids=None):
    if 'coco' in data_name:
        dataset = CocoDataset(root=root, json=json, vocab=vocab, transform=transform, ids=ids)
    elif 'f8k' in data_name or 'f30k' in data_name:
        dataset = FlickrDataset(root=root, split=split, json=json, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab, opt)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name in ['val', 'test']:
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    transform = transforms.Compose(t_list + [transforms.ToTensor(), normalizer])
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt, batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, 'dev', vocab, opt, batch_size, False, workers)
    else:
        roots, ids = get_paths(dpath, data_name, opt.use_restval)
        transform = get_transform(data_name, 'train', opt)
        train_loader = get_loader_single(opt.data_name, 'train', roots['train']['img'],
                                         roots['train']['cap'], vocab, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True, num_workers=workers)
        transform = get_transform(data_name, 'val', opt)
        val_loader = get_loader_single(opt.data_name, 'val', roots['val']['img'],
                                       roots['val']['cap'], vocab, transform, ids=ids['val'],
                                       batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt, batch_size, False, workers)
    else:
        roots, ids = get_paths(dpath, data_name, opt.use_restval)
        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name, roots[split_name]['img'],
                                        roots[split_name]['cap'], vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False, num_workers=workers)
    return test_loader
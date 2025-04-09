import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO
import json
import argparse
import os

# Dataset annotations mapping
annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'coco': ['annotations/captions_train2014.json', 'annotations/captions_val2014.json'],
    'f8k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    '10crop_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f8k': ['dataset_flickr8k.json'],
    'f30k': ['dataset_flickr30k.json'],
}


class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)


def from_coco_json(path):
    coco = COCO(path)
    captions = [str(coco.anns[idx]['caption']) for idx in coco.anns.keys()]
    return captions


def from_flickr_json(path):
    dataset = json.load(open(path, 'r'))['images']
    captions = [str(x['raw']) for d in dataset for x in d['sentences']]
    return captions


def from_txt(txt):
    with open(txt, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f]
    return captions


def build_vocab(data_path, data_name, jsons, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for path in jsons[data_name]:
        full_path = os.path.join(data_path, data_name, path)
        if data_name == 'coco':
            captions = from_coco_json(full_path)
        elif data_name in ['f8k', 'f30k']:
            captions = from_flickr_json(full_path)
        else:
            captions = from_txt(full_path)
        
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            if i % 1000 == 0:
                print(f"[{i}/{len(captions)}] tokenized the captions.")

    # Filter words by threshold
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create vocabulary
    vocab = Vocabulary()
    for special_token in ['<pad>', '<start>', '<end>', '<unk>']:
        vocab.add_word(special_token)
    for word in words:
        vocab.add_word(word)
    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, jsons=annotations, threshold=4)
    os.makedirs('./vocab', exist_ok=True)
    vocab_file = f'./vocab/{data_name}_vocab.pkl'
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved vocabulary file to {vocab_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Vocabulary for VSRN')
    parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/', help='Path to dataset directory')
    parser.add_argument('--data_name', default='coco', help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)
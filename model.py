import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from GCN_lib.Rs_GCN import Rs_GCN

import opts
import misc.utils as utils
import torch.optim as optim

from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel


def l2norm(X):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    return torch.div(X, norm.clamp(min=1e-8))  # Avoid division by zero


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False, use_txt_emb=True):
    if data_name.endswith('_precomp'):
        if use_txt_emb:
            img_enc = EncoderImagePrecompAttn(
                img_dim, embed_size, data_name, use_abs, no_imgnorm)
        else:
            img_enc = EncoderImagePrecomp(
                img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)
    return img_enc


class EncoderImageFull(nn.Module):
    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.cnn = self.get_cnn(cnn_type, True)
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier[-1].in_features, embed_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
            self.cnn.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        print(f"=> {'using pre-trained' if pretrained else 'creating'} model '{arch}'")
        model = getattr(models, arch)(weights='DEFAULT' if pretrained else None)
        return nn.DataParallel(model)

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        nn.init.uniform_(self.fc.weight, -r, r)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        features = self.cnn(images)
        features = l2norm(features)
        features = self.fc(features)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        nn.init.uniform_(self.fc.weight, -r, r)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        features = self.fc(images)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features


class EncoderImagePrecompAttn(nn.Module):
    def __init__(self, img_dim, embed_size, data_name, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.data_name = data_name

        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        if self.data_name == 'f30k_precomp':
            self.bn = nn.BatchNorm1d(embed_size)

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        nn.init.uniform_(self.fc.weight, -r, r)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        fc_img_emd = self.fc(images)
        if self.data_name != 'f30k_precomp':
            fc_img_emd = l2norm(fc_img_emd)

        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = l2norm(GCN_img_emd)

        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)
        features = hidden_state[0]

        if self.data_name == 'f30k_precomp':
            features = self.bn(features)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features, GCN_img_emd


class EncoderText(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    def forward(self, x, lengths):
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.tensor(lengths, device=x.device).view(-1, 1, 1).expand(x.size(0), 1, self.embed_size) - 1
        out = torch.gather(padded[0], 1, I).squeeze(1)
        out = l2norm(out)
        if self.use_abs:
            out = torch.abs(out)
        return out


def cosine_sim(im, s):
    return im @ s.t()


def order_sim(im, s):
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = order_sim if measure == 'order' else cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0), device=scores.device).bool()
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class VSRN(nn.Module):
    def __init__(self, opt):
        super(VSRN, self).__init__()
        self.grad_clip = opt.grad_clip
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type, opt.use_abs, opt.no_imgnorm).to(self.device)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, opt.use_abs).to(self.device)

        self.encoder = EncoderRNN(
            opt.dim_vid, opt.dim_hidden, bidirectional=opt.bidirectional,
            input_dropout_p=opt.input_dropout_p, rnn_cell=opt.rnn_type,
            rnn_dropout_p=opt.rnn_dropout_p).to(self.device)
        self.decoder = DecoderRNN(
            opt.vocab_size, opt.max_len, opt.dim_hidden, opt.dim_word,
            input_dropout_p=opt.input_dropout_p, rnn_cell=opt.rnn_type,
            rnn_dropout_p=opt.rnn_dropout_p, bidirectional=opt.bidirectional).to(self.device)
        self.caption_model = S2VTAttModel(self.encoder, self.decoder).to(self.device)

        self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.criterion = ContrastiveLoss(margin=opt.margin, measure=opt.measure,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters()) + \
                 list(self.decoder.parameters()) + list(self.encoder.parameters()) + \
                 list(self.caption_model.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params
        self.optimizer = optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        # self.logger = utils.Logger()  # 假设存在此工具类

    def calculate_caption_loss(self, fc_feats, labels, masks):
        labels = labels.to(self.device)
        masks = masks.to(self.device)
        seq_probs, _ = self.caption_model(fc_feats, labels, 'train')
        loss = self.crit(seq_probs, labels[:, 1:], masks[:, 1:])
        return loss

    def state_dict(self):
        return [self.img_enc.state_dict(), self.txt_enc.state_dict()]

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """确保所有组件都处于训练模式"""
        # 首先将整个模型设置为训练模式
        self.train()
        
        # 确保主要组件处于训练模式
        self.img_enc.train()
        self.txt_enc.train()
        self.caption_model.train()
        
        # 特别确保所有RNN组件处于训练模式
        self.encoder.train()
        self.decoder.train()
        
        # 递归确保EncoderImagePrecompAttn中的img_rnn处于训练模式
        if hasattr(self.img_enc, 'img_rnn'):
            self.img_enc.img_rnn.train()
        
        # 确保EncoderText中的rnn处于训练模式
        if hasattr(self.txt_enc, 'rnn'):
            self.txt_enc.rnn.train()
        
        # 打印训练状态以进行调试
        print("Model training mode:", self.training)
        print("img_enc training mode:", self.img_enc.training)
        print("txt_enc training mode:", self.txt_enc.training)
        print("caption_model training mode:", self.caption_model.training)


    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.caption_model.eval()

    def forward_emb(self, images, captions, lengths):
        images = images.to(self.device)
        captions = captions.to(self.device)
        cap_emb = self.txt_enc(captions, lengths)
        img_emb, GCN_img_emd = self.img_enc(images)
        return img_emb, cap_emb, GCN_img_emd

    def forward_loss(self, img_emb, cap_emb):
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le_retrieval', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids, caption_labels, caption_masks, *args):
        self.train()
        self.img_enc.train()
        self.txt_enc.train()
        self.caption_model.train()
        self.encoder.train()
        self.decoder.train()
        
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb, GCN_img_emd = self.forward_emb(images, captions, lengths)
        self.optimizer.zero_grad()

        caption_loss = self.calculate_caption_loss(GCN_img_emd, caption_labels, caption_masks)
        retrieval_loss = self.forward_loss(img_emb, cap_emb)
        loss = retrieval_loss + caption_loss

        self.logger.update('Le_caption', caption_loss.item(), img_emb.size(0))
        self.logger.update('Le', loss.item(), img_emb.size(0))

        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
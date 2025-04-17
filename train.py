import pickle
import os
import time
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

import data
from vocab import Vocabulary
from model import VSRN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
import logging
import argparse
import wandb

def main():
    parser = argparse.ArgumentParser(description='VSRN Training Script')
    parser.add_argument('--data_path', default='/data', help='path to datasets')
    parser.add_argument('--data_name', default='precomp', help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=2048, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2.0, type=float, help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int, help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19', help='The CNN used for image encoder (e.g. vgg19, resnet152)')
    parser.add_argument('--use_restval', action='store_true', help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true', help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true', help='Ensure training is always done in train mode.')

    # Caption parameters
    parser.add_argument('--dim_vid', type=int, default=2048, help='dim of features of video frames')
    parser.add_argument('--dim_hidden', type=int, default=512, help='size of the rnn hidden layer')
    parser.add_argument('--bidirectional', type=int, default=0, help='0 for disable, 1 for enable.')
    parser.add_argument('--input_dropout_p', type=float, default=0.2, help='dropout in the Language Model RNN')
    parser.add_argument('--rnn_type', type=str, default='gru', help='lstm or gru')
    parser.add_argument('--rnn_dropout_p', type=float, default=0.5, help='dropout in the Language Model RNN')
    parser.add_argument('--dim_word', type=int, default=300, help='encoding size of each token')
    parser.add_argument('--max_len', type=int, default=60, help='max length of captions (including <sos>,<eos>)')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', default='VSRN', help='Project name for Weights & Biases')
    parser.add_argument('--wandb_name', default='VSRN', help='Run name for Weights & Biases')


    opt = parser.parse_args()
    print(opt)
    if opt.use_wandb:
        wandb.init(project=opt.wandb_project, name=opt.wandb_name, config=vars(opt))

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = SummaryWriter(log_dir=opt.logger_name, flush_secs=5)

    # Load Vocabulary
    with open(os.path.join(opt.vocab_path, f'{opt.data_name}_vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = VSRN(opt)
    if opt.use_wandb:
        wandb.watch(model, log="all")

    # Resume from checkpoint if provided
    start_epoch = 0
    best_rsum = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume, map_location=model.device)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            print(f"=> loaded checkpoint '{opt.resume}' (epoch {start_epoch}, best_rsum {best_rsum})")
            validate(opt, val_loader, model)
        else:
            print(f"=> no checkpoint found at '{opt.resume}'")

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)
        best_rsum = train(opt, train_loader, model, epoch, val_loader, best_rsum)

        rsum = validate(opt, val_loader, model)
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader, best_rsum):
    # AverageMeter()是用于计算和存储平均值的辅助类，根据训练日志。
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    tb_logger = SummaryWriter(log_dir=opt.logger_name)

    # 确保模型处于训练模式
    model.train_start()
    end = time.time()

    for i, train_data in enumerate(train_loader):
        if opt.reset_train:
            model.train_start()

        data_time.update(time.time() - end)
        model.logger = train_logger
        # 训练图像-文本的联合嵌入(joint embedding)，学习图像和文本的联合嵌入空间，确保语义相关的图像-文本对在嵌入空间距离更近
        # 实现视觉语义推理的训练过程
        model.train_emb(*train_data)

        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:
            logging.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'{str(model.logger)}\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})'
            )

        tb_logger.add_scalar('epoch', epoch, global_step=model.Eiters)
        tb_logger.add_scalar('step', i, global_step=model.Eiters)
        tb_logger.add_scalar('batch_time', batch_time.val, global_step=model.Eiters)
        tb_logger.add_scalar('data_time', data_time.val, global_step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        if opt.use_wandb:
                # 创建一个字典包含所有要记录的指标
                metrics = {
                    'train/epoch': epoch,
                    'train/step': i,
                    'train/batch_time': batch_time.val,
                    'train/data_time': data_time.val,
                }
                
                # 从model.logger获取训练指标并添加到metrics
                for key, value in model.logger.meters.items():
                    metrics[f'train/{key}'] = value.val
                
                # 记录到wandb
                wandb.log(metrics, step=model.Eiters)

        # evaluation
        if model.Eiters % opt.val_step == 0:
            rsum = validate(opt, val_loader, model)
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, prefix=opt.logger_name + '/')

    return best_rsum


def validate(opt, val_loader, model):
    tb_logger = SummaryWriter(log_dir=opt.logger_name)
    img_embs, cap_embs = encode_data(model, val_loader)
    
    # Get metrics for image-to-text
    metrics = i2t(img_embs, cap_embs, measure=opt.measure)
    if isinstance(metrics, tuple) and len(metrics) == 2:
        metrics = metrics[0]  # Take only the metrics, not the ranks
    r1, r5, r10, medr, meanr = metrics
    logging.info(f"Image to text: {r1:.3f}, {r5:.3f}, {r10:.3f}, {medr:.3f}, {meanr:.3f}")
    logging.info(f"metrics: {metrics}")
    
    # Get metrics for text-to-image
    metrics = t2i(img_embs, cap_embs, measure=opt.measure)
    if isinstance(metrics, tuple) and len(metrics) == 2:
        metrics = metrics[0]  # Take only the metrics, not the ranks
    r1i, r5i, r10i, medri, meanr = metrics
    logging.info(f"Text to image: {r1i:.3f}, {r5i:.3f}, {r10i:.3f}, {medri:.3f}, {meanr:.3f}")
    
    currscore = r1 + r5 + r1i + r5i

    tb_logger.add_scalar('r1', r1, global_step=model.Eiters)
    tb_logger.add_scalar('r5', r5, global_step=model.Eiters)
    tb_logger.add_scalar('r10', r10, global_step=model.Eiters)
    tb_logger.add_scalar('medr', medr, global_step=model.Eiters)
    tb_logger.add_scalar('meanr', meanr, global_step=model.Eiters)
    tb_logger.add_scalar('r1i', r1i, global_step=model.Eiters)
    tb_logger.add_scalar('r5i', r5i, global_step=model.Eiters)
    tb_logger.add_scalar('r10i', r10i, global_step=model.Eiters)
    tb_logger.add_scalar('medri', medri, global_step=model.Eiters)
    tb_logger.add_scalar('meanr', meanr, global_step=model.Eiters)
    tb_logger.add_scalar('rsum', currscore, global_step=model.Eiters)


    if opt.use_wandb:
        if hasattr(opt, 'use_wandb') and opt.use_wandb:
            wandb_metrics = {
                'val/r1': r1,
                'val/r5': r5,
                'val/r10': r10,
                'val/medr': medr,
                'val/meanr': meanr,
                'val/r1i': r1i,
                'val/r5i': r5i,
                'val/r10i': r10i,
                'val/medri': medri,
                'val/rsum': currscore
            }
            wandb.log(wandb_metrics, step=model.Eiters)
    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k * (100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
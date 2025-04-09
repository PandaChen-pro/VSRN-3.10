import argparse


def parse_opt():
    parser = argparse.ArgumentParser(description='VSRN Training Arguments')

    # Data input settings
    parser.add_argument('--data_path', type=str, default='data',
                        help='path to the dataset directory')
    parser.add_argument('--data_name', type=str, default='coco_precomp',
                        help='dataset name (coco, f8k, f30k, coco_precomp, f8k_precomp, f30k_precomp)')
    parser.add_argument('--vocab_path', type=str, default='vocab',
                        help='path to the vocabulary directory')
    parser.add_argument('--input_json', type=str, default='data/videodatainfo_2017.json',
                        help='path to the json file containing video info')
    parser.add_argument('--info_json', type=str, default='data/info.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--caption_json', type=str, default='data/caption.json',
                        help='path to the processed video caption json')
    parser.add_argument('--feats_dir', nargs='*', type=str, default=['data/feats/resnet152/'],
                        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--c3d_feats_dir', type=str, default='data/c3d_feats',
                        help='path to the directory containing C3D features')
    parser.add_argument('--with_c3d', type=int, default=0,
                        help='whether to use C3D features (0 = no, 1 = yes)')
    parser.add_argument('--cached_tokens', type=str, default='msr-all-idxs',
                        help='cached token file for calculating CIDEr score during self-critical training')

    # Model settings (VSRN-specific)
    parser.add_argument('--img_dim', type=int, default=2048,
                        help='dimension of image features')
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='size of the joint embedding space')
    parser.add_argument('--word_dim', type=int, default=300,
                        help='dimension of word embeddings')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='size of the vocabulary')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the text RNN')
    parser.add_argument('--finetune', type=int, default=0,
                        help='whether to finetune the CNN (0 = no, 1 = yes)')
    parser.add_argument('--cnn_type', type=str, default='vgg19',
                        help='type of CNN to use (vgg19, resnet152, etc.)')
    parser.add_argument('--use_abs', type=int, default=0,
                        help='whether to use absolute value in embeddings (0 = no, 1 = yes)')
    parser.add_argument('--no_imgnorm', type=int, default=0,
                        help='whether to skip image normalization (0 = normalize, 1 = no)')

    # Captioning model settings (from original S2VT)
    parser.add_argument('--model', type=str, default='S2VTModel',
                        help='model to use for captioning (S2VTModel, etc.)')
    parser.add_argument('--max_len', type=int, default=28,
                        help='max length of captions (including <sos>, <eos>)')
    parser.add_argument('--bidirectional', type=int, default=0,
                        help='whether encoder/decoder is bidirectional (0 = no, 1 = yes)')
    parser.add_argument('--dim_hidden', type=int, default=512,
                        help='size of the RNN hidden layer')
    parser.add_argument('--input_dropout_p', type=float, default=0.2,
                        help='dropout strength in the language model RNN input')
    parser.add_argument('--rnn_type', type=str, default='gru',
                        help='type of RNN (lstm or gru)')
    parser.add_argument('--rnn_dropout_p', type=float, default=0.5,
                        help='dropout strength in the language model RNN')
    parser.add_argument('--dim_word', type=int, default=512,
                        help='encoding size of each token in the vocabulary')
    parser.add_argument('--dim_vid', type=int, default=2048,
                        help='dimension of video frame features')

    # Optimization settings
    parser.add_argument('--epochs', type=int, default=6001,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip gradients at this value')
    parser.add_argument('--self_crit_after', type=int, default=-1,
                        help='epoch to start self-critical training (-1 = disable)')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='initial learning rate')
    parser.add_argument('--learning_rate_decay_every', type=int, default=200,
                        help='decay learning rate every N epochs')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                        help='learning rate decay factor')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for Adam optimizer')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta for Adam optimizer')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='strength of weight regularization')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='margin for contrastive loss')
    parser.add_argument('--measure', type=str, default='cosine',
                        help='similarity measure (cosine or order)')
    parser.add_argument('--max_violation', type=int, default=1,
                        help='whether to use max violation in contrastive loss (0 = no, 1 = yes)')

    # Checkpoint settings
    parser.add_argument('--save_checkpoint_every', type=int, default=50,
                        help='how often to save a model checkpoint (in epochs)')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                        help='directory to store checkpointed models')

    # Hardware settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device number')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
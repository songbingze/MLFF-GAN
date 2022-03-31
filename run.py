import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from experiment import Experiment

import os
import faulthandler

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
faulthandler.enable()

"""
jsub -q qgpu -J ganstfm -e error.txt -o output.txt python run.py --lr 2e-4 --num_workers 36 --batch_size 36 --epochs 500 --cuda --ngpu 2 --image_size 2040 1720 --save_dir out --data_dir data

jsub -q qgpu -J ganstfm -e error.txt -o output.txt python run.py --lr 2e-4 --num_workers 36 --batch_size 36 --epochs 500 --cuda --ngpu 2 --image_size 2720 3200 --patch_size 1360 1600 --save_dir out --data_dir data
"""

def str2bool(str):
    return True if str.lower() == 'true' else False

# 获取模型运行时必须的一些参数
parser = argparse.ArgumentParser(description='Acquire some parameters for fusion restore')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=4, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('savepath'),
                    help='the output directory')

# 获取对输入数据进行预处理时的一些参数
parser.add_argument('--data_dir', type=Path,  default=r'CIA',
                    help='the training data directory')
parser.add_argument('--image_size', type=int, nargs='+',default=[1792, 1280],
                    help='the image size (height, width)')
parser.add_argument('--patch_stride', type=int, nargs='+', default=200,
                    help='the patch stride for training')
parser.add_argument('--patch_size', type=int, nargs='+', default=[256 * 4, 256 * 3],
                    help='the patch size for prediction')
parser.add_argument('--cut_num', type=int, nargs='+', default=[3, 3],
                    help='cut the image num about the patch to keep the repeated part on the patch edge for removing the uncertainty')
parser.add_argument('--ifAdaIN', type=str2bool, default='True',
                    help='use AdaIN?')
parser.add_argument('--ifAttention', type=str2bool, default='True',
                    help='use Attention?')
parser.add_argument('--ifTwoInput', type=str2bool, default='False',
                    help='use two-inputs mode or three-inputs mode?')
parser.add_argument('--ifAmplificat', type=str2bool, default='True',
                    help='use Amplificat?')
parser.add_argument('--a', type=float, default=1e-2,
                    help='$alpha$ of the G loss')
parser.add_argument('--b', type=float, default=1,
                    help='$beta$ of the G loss')
parser.add_argument('--c', type=float, default=1,
                    help='$gamma$ of the G loss')
parser.add_argument('--d', type=float, default=1,
                    help='$delta$ of the G loss')


opt = parser.parse_args()

torch.manual_seed(2020)
opt.cuda = True
if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    torch.cuda.manual_seed_all(2020)
    cudnn.benchmark = True
    cudnn.deterministic = True


if __name__ == '__main__':
    experiment = Experiment(opt)
    train_dir = opt.data_dir / 'train'
    val_dir = opt.data_dir / 'val'
    test_dir = val_dir

    if opt.epochs > 0:
        if opt.epochs > 0:
            experiment.train(train_dir, val_dir,
                             opt.patch_stride, opt.batch_size,
                             num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(test_dir,opt.image_size, opt.patch_size, num_workers=opt.num_workers)

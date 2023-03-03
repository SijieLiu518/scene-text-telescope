import os
import yaml
import argparse
import torch
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR

def main(config, args):
    Mission = TextSR(config, args)
    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()

if __name__ == '__main__':
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tbsrn', choices=['tbsrn', 'tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn', 'chatgpt_sr', 'tbsrn_seg_emb'])
    parser.add_argument('--text_focus', action='store_true')
    parser.add_argument('--exp_name', required=True, help='Type your experiment name')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='./dataset/mydata/test')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--rec', default='crnn', choices=['crnn', 'moran', 'aster'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--cp_loss', action='store_true', default=False, help='')
    parser.add_argument('--wgan_loss', action='store_true', default=False, help='')
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)

from chemprop.train import cross_validate
from argparse import ArgumentParser
import torch


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int,
                        default = 40,
                        help='torch random seed')
    parser.add_argument('--data_path', type=str,
                        default= "./input/train.csv",
                        help='Path to CSV input file containing training data'
                       )
    parser.add_argument('--save_dir', type=str,
                        default= "./output/train.csv",
                        help='Path to CSV result(output) file containing training data'
                       )
    parser.add_argument('--num_folds', type=int,
                        default = 4,
                        help='num fold')
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--features_path', type=str,
                        help='features_path: (?)')
    parser.add_argument('--features_generator', type=str, nargs='+',
                        help='features_generator: morgan, morgan_count ...')
    parser.add_argument('--max_data_size', type=int,
                        help='max_data_size')


    args = parser.parse_args()
    cross_validate(args)
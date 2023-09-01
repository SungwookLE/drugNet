from chemprop.train import cross_validate
from argparse import ArgumentParser
import torch


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int,
                        default = 1213,
                        help='torch random seed')
    parser.add_argument('--data_path', type=str,
                        default= "./input/train.csv",
                        help='Path to CSV input file containing training data'
                       )
    parser.add_argument('--save_dir', type=str,
                        default= "./output",
                        help='Path to result(output model.pt)'
                       )
    parser.add_argument('--num_folds', type=int,
                        default = 4,
                        help='num fold')
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--features_path', type=str,
                        help='features_path: FingerPrint를 따로 저장해둔 파일이 있으면 입력')
    parser.add_argument('--features_generator', type=str, nargs='+',
                        help='features_generator: morgan, morgan_count ...')
    parser.add_argument('--max_data_size', type=int,
                        help='max_data_size')
    parser.add_argument('--separate_test_path', type=str,
                        default = "./input/test.csv",
                        help='test.csv')
    parser.add_argument('--separate_test_features_path', type=str,
                        default = None)
    parser.add_argument('--dataset_type', type=str,
                        default = "regression",
                        help='MLM, HLM Prediction as Regression')
    parser.add_argument('--metric', type=str,
                        default = "rmse",
                        help='Regression Loss Function')
    parser.add_argument('--cuda', action='store_true',default=False,
                        help='Turn on cuda')
    parser.add_argument('--split_type', type=str,
                        default = "random", help="How to split the data")
    parser.add_argument('--features_scaling', action='store_true', default=False,
                        help='Features Scaler')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='ensemble_size')
    parser.add_argument('--checkpoint_paths', type=str, default=None,
                        help='checkpoint_paths')
    parser.add_argument('--activation', type=str, default='ReLU',
                            help='activation')
    parser.add_argument('--atom_messages', type=bool, default=False,
                            help='atom_messages')
    parser.add_argument('--batch_size', type=int, default=100,
                            help='batch_size')
    parser.add_argument('--bias', type=bool, default=False,
                            help='bias')
    parser.add_argument('--hidden_size', type=int, default=8,
                            help='hidden_size')
    parser.add_argument('--depth', type=int, default=5,
                            help='depth')
    parser.add_argument('--dropout', type=float, default=0.05,
                            help='dropout')
    parser.add_argument('--undirected', type=bool, default=False,
                            help='undirected')
    parser.add_argument('--features_only', type=bool, default=False,
                            help='features_only')
    parser.add_argument('--use_input_features', action='store_true', default=False,
                            help='use_input_features: feature concat을 결정하는 파라미터')
    parser.add_argument('--init_lr', type=float, default=0.0001,
                            help='init_lr')
    parser.add_argument('--final_lr', type=float, default=0.0001,
                            help='final_lr')
    parser.add_argument('--max_lr', type=float, default=0.001,
                            help='max_lr')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                            help='warmup_epochs')
    parser.add_argument('--epochs', type=int, default=50,
                            help='epochs')
    parser.add_argument('--num_lrs', type=int, default=1,
                            help='num_lrs')
    parser.add_argument('--minimize_score', type=bool, default=True,
                            help='minimize_score')
    parser.add_argument('--no_cache', type=bool, default=False,
                            help='no_cache')
    parser.add_argument('--log_frequency', type=int, default=10,
                            help='log_frequency')
    parser.add_argument('--show_individual_scores', type=bool, default=False,
                            help='show_individual_scores')
    parser.add_argument('--ffn_num_layers', type=int, default=1,
                            help='ffn_num_layers')
    parser.add_argument('--ffn_hidden_size', type=int, default=64,
                            help='ffn_hidden_size')

    args = parser.parse_args()
    cross_validate(args)
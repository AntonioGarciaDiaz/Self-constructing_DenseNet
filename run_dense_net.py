import argparse
import os

from models.dense_net import DenseNet
from data_providers.utils import get_data_provider_by_name

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_params_cifar = {
    'batch_size': 64,
    'max_n_epochs': 80,  # default was 300
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,  # epochs * 0.5, default was 150
    'reduce_lr_epoch_2': 30,  # epochs * 0.75, default was 225
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'max_n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet',
        help='Choice of model to use (use bottleneck + compression or not).')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 40],
        default=12,
        help='Growth rate (number of new convolutions per dense layer), '
             'restricted to choices in paper.')
    parser.add_argument(
        '--dataset', '-ds', type=str,
        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],
        default='C10',
        help='Choice of dataset to use.')
    parser.add_argument(
        '--layer_num_list', '-lnl',
        type=str, default='1', metavar='',
        help='List of the number of layers in each block, separated by comas'
             ' (e.g. \'12,12,12\', default: 1 block with 1 layer)'
             'WARNING: in BC models, each layer is preceded by a bottleneck.')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',
        help="Keeping probability, for dropout")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
        help='Weight decay, for optimizer (default: %(default)s).')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s).')
    parser.add_argument(
        '--reduction', '-red', '-theta', type=float, default=0.5, metavar='',
        help='Reduction (theta) at transition layer, for DenseNets-BC models.')

    parser.add_argument(
        '--self-construct', dest='should_self_construct', action='store_true',
        help='Apply a self-constructing algorithm for modifying'
             'the network\'s architecture during training.')
    parser.add_argument(
        '--no-self-construct', dest='should_self_construct',
        action='store_false',
        help='Do not apply a self-constructing algorithm.')
    parser.set_defaults(should_self_construct=True)

    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs.')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs.')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
        '--feature_period', '-fp', type=int, default=5,
        help='Number of epochs between each measurement of feature values.')
    parser.add_argument(
        '--feature-logs', dest='should_save_ft_logs', action='store_true',
        help='Record the evolution of feature values in a CSV log.')
    parser.add_argument(
        '--no-feature-logs', dest='should_save_ft_logs', action='store_false',
        help='Do not record feature values in a CSV log.')
    parser.set_defaults(should_save_ft_logs=True)
    parser.add_argument(
        '--kernel-features', dest='check_kernel_features', action='store_true',
        help='Measure feature values from convolution kernels'
             '(e.g. the mean and std of a filter\'s kernel values).')
    parser.add_argument(
        '--no-kernel-features', dest='check_kernel_features',
        action='store_false',
        help='Do not measure feature values from kernels.')
    parser.set_defaults(check_kernel_features=True)

    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training.')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training.')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
        '--images', dest='should_save_images', action='store_true',
        help='Produce and save image files (e.g. representing kernel states).')
    parser.add_argument(
        '--no-images', dest='should_save_images', action='store_false',
        help='Do not produce and save image files.')
    parser.set_defaults(should_save_images=False)

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if they exist.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if they exist.')

    parser.add_argument(
        '--num_inter_threads', '-inter', type=int, default=1, metavar='',
        help='Number of inter-operation CPU threads '
             '(for paralellizing the inference/testing phase).')
    parser.add_argument(
        '--num_intra_threads', '-intra', type=int, default=128, metavar='',
        help='Number of intra-operation CPU threads '
             '(for paralellizing the inference/testing phase).')

    parser.set_defaults(renew_logs=True)

    args = parser.parse_args()

    if not args.keep_prob:
        if args.dataset in ['C10', 'C100', 'SVHN']:
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0
    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 1.0
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Prepare training data...")
    data_provider = get_data_provider_by_name(args.dataset, train_params)
    print("Initialize the model...")
    model = DenseNet(data_provider=data_provider, **model_params)
    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=200)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))

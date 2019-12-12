import argparse
import datetime
import importlib
import os
import sys
import warnings

from mxnet import npx


npx.set_np()
npx.random.seed(28)
sys.setrecursionlimit(2000)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
warnings.simplefilter("ignore")


def main():
    print(datetime.datetime.now())

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--app', type=str, help='The app to use.')
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        choices=['all', 'train', 'test'],
        default='all',
        help='The mode to run. `all` will train and test epochs alternately. \
        `train` and `test` will train and test 1 epoch, respectively.'
    )
    parser.add_argument(
        '-c', '--config', type=int, default=1, help='Which config file to use.'
    )
    args = parser.parse_args()

    config = importlib.import_module(
        '{}.confs.config_{}'.format(args.app, args.config)
    )

    print('==========>> build dataset')
    dataset_module = importlib.import_module(
        '{}.data.dataset'.format(args.app)
    )
    dataset = {
        'train': dataset_module.Dataset(config, 'train'),
        'test': dataset_module.Dataset(config, 'test')
    }

    print('==========>> build model')
    model_module = importlib.import_module('{}.model.model'.format(args.app))
    model = model_module.Model(config)

    print('==========>> build trainer')
    manager_module = importlib.import_module(
        '{}.manager.manager'.format(args.app)
    )
    manager = manager_module.Manager(model, dataset, config)

    print('==========>> start to run model')
    if args.mode == 'all':
        manager.train(test=True)
    elif args.mode == 'train':
        manager.train_epoch()
    elif args.mode == 'test':
        manager.test_epoch()


if __name__ == '__main__':
    main()

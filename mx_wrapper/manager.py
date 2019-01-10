import os
import sys
import time

import mxnet as mx
from mxnet import autograd, lr_scheduler, nd
from mxnet.gluon import nn, trainer
from mxnet.gluon.data import dataloader
import numpy as np

from . import utils


class Manager:
    """The train/test manager.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging.

    Interfaces:
        compute_loss(self, outputs, labels)
        setup_handler(self, epoch, mode, num_batch, ds_size)

    """

    def __init__(self, model, dataset, config):
        """
        Parameters
        ----------
        model: mxnet.gluon.nn.Block or mxnet.gluon.nn.HybridBlock
        dataset: dict
            A dict with two keys: `train` and `test`. Each value is an
            `mxnet.gluon.data.Dataset` of the corresponding mode.
        config: module
            The global config module.
        """
        self.model = model
        self.dataset = dataset
        self.config = config

        self.ctx = [mx.cpu()]
        if config.GPUS and mx.test_utils.list_gpus():
            self.ctx = [mx.gpu(i) for i in config.GPUS]

        self.latest_state = utils.init_model(
            self.model,
            self.config.PARAM_DIR,
            self.config.PARAM_PREFIX,
            self.config.PARAM_INDEX,
            self.config.PRETRAIN_PATH,
            ctx=self.ctx
        )
        if isinstance(self.model, nn.HybridBlock):
            self.model.hybridize()

        self.trainer = self.create_trainer()

    def train(self, test=True):
        for _ in range(self.latest_state + 1, self.config.MAX_EPOCHS):
            self.train_epoch()
            if test:
                self.test_epoch()

    def train_epoch(self):
        epoch = self.latest_state + 1
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'train')
        t = time.time()
        print('train 1 epoch in {}\n'.format(utils.parse_time(t - s)))

        if (
            not self.config.SAVE_EPOCH_FREQ or
            epoch % self.config.SAVE_EPOCH_FREQ == 0 or
            epoch == self.config.MAX_EPOCHS
        ):
            self.model.save_parameters(
                os.path.join(
                    self.config.PARAM_DIR,
                    '{}.params-{:04d}'.format(self.config.PARAM_PREFIX, epoch)
                )
            )
        self.latest_state = epoch

    def test_epoch(self):
        epoch = self.latest_state
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'test')
        t = time.time()
        print('test 1 epoch in {}\n\n\n'.format(utils.parse_time(t - s)))

    def _process_epoch(self, epoch, mode):
        color_code = '\033[1;32m' if sys.platform != 'win32' else ''
        end_color_code = '\033[0m' if sys.platform != 'win32' else ''
        print(
            color_code + \
            '{}: epoch {:3d}/{:3d}'.format(mode, epoch, self.config.MAX_EPOCHS) + \
            end_color_code
        )

        loader = self.create_dataloader(mode)
        handler = self.create_handler(mode=mode, num_batch=len(loader))

        for i, raw_data in enumerate(loader):
            bs = raw_data[0].shape[0]
            gathered_outputs = []
            gathered_losses = []
            losses = []
            tick = time.time()
            splited_data = utils.split_and_load(raw_data, self.ctx)

            if mode == 'train':
                autograd.set_training(True)
                autograd.set_recording(True)
            elif mode == 'test':
                autograd.set_training(False)
                autograd.set_recording(False)

            for data in splited_data:
                inputs, labels = self.parse_data(data, mode)
                outputs = self.parse_output(self.model(*inputs), mode)
                gathered_outputs.append(outputs)
                loss = self.compute_loss(outputs, labels)
                gathered_losses.append(loss)
                if mode == 'train':
                    losses.extend(loss)

            autograd.set_training(False)
            autograd.set_recording(False)

            if mode == 'train':
                autograd.backward(losses)
                self.trainer.step(bs)

            handler.cleanup_batch(
                raw_data, gathered_outputs, gathered_losses, i, tick
            )
        handler.cleanup_epoch()

    def create_trainer(self):
        train_params = utils.collect_train_params(self.model, self.config)
        scheduler = self.create_lr_scheduler()
        optimizer_params = self.config.OPT_PARAMS.copy()
        if isinstance(scheduler, lr_scheduler.LRScheduler):
            optimizer_params.update({'lr_scheduler': scheduler})
        return trainer.Trainer(
            train_params, self.config.OPTIMIZER, optimizer_params
        )

    def create_lr_scheduler(self):
        """Create a learning rate scheduler.
        Default: Use no scheduler.

        Returns
        -------
        mxnet.lr_scheduler.LRScheduler
        """
        return None

    def create_dataloader(self, mode):
        """Set up the dataloader for the current mode for each epoch.

        Parameters
        ----------
        mode: str
            `train` or `test`.

        Returns
        -------
        mxnet.gluon.data.Dataloader
        """
        dataset = self.dataset[mode]
        loader = dataloader.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE[mode],
            shuffle=True,
            batchify_fn=dataset.batchify_fn,
            num_workers=self.config.NUM_WORKERS
        )
        return loader

    def parse_data(self, data, mode):
        """Separate inputs and labels.

        Parameters
        ----------
        data: list of mxnet.nd.NDArray
            [input_1[, input_2, ...], label_1[, label_2, ...[, aux_1, ...]]]
        mode: str
            `train` or `test`.

        Returns
        -------
        tuple
            The first element is the list of inputs,
            the second the list of labels.
        """
        return data[:-1], [data[-1]]

    def parse_output(self, outputs, mode):
        """Make the outputs suitable for computing loss.

        Parameters
        ----------
        outputs: mxnet.nd.NDArray or list of mxnet.nd.NDArray
        mode: str
            `train` or `test`.

        Returns
        -------
        list of mxnet.nd.NDArray
        """
        return [outputs] if isinstance(outputs, nd.NDArray) else outputs

    def compute_loss(self, outputs, labels):
        """Compute the loss.

        Parameters
        ----------
        outputs: list of mxnet.nd.NDArray
        labels: list of mxnet.nd.NDArray

        Returns
        -------
        list:
            Contain different losses. Even if there's only 1 loss, should wrap
            it in a list.
        """
        raise NotImplementedError

    def create_handler(self, mode, num_batch):
        """Sets up a handler to perform logging or postprocessing after each
        batch and each epoch. A handler should have the following methods:
            `cleanup_batch(self, data, outputs, losses, batch, tick)`
            `cleanup_epoch(self)`
        Specifically, `batch` is the current batch index and `hz` is the speed
        for processing the current batch.

        Parameters
        ----------
        mode: str
            `train` or `test`.
        num_batch: int
            Number of batches.

        Returns
        -------
        Object
            The handler.
        """
        raise NotImplementedError

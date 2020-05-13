import os
import sys
import time

import mxnet as mx
from mxnet import autograd, lr_scheduler, nd, util
from mxnet.gluon import nn, trainer
from mxnet.gluon.data import dataloader

from . import esc_seq, utils


class Manager:
    """The train/test manager.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging.

    Interfaces:
        compute_loss(self, outputs, labels)
        create_handler(self, mode, num_batch)

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
            ctx=self.ctx,
        )
        if isinstance(self.model, nn.HybridBlock):
            print("Hybridize model..")
            self.model.hybridize()

        self.trainer = self.create_trainer()

    def export_model(self):
        if not isinstance(self.model, nn.HybridBlock):
            raise ValueError(
                "Expected a HybridBlock but the model seems not one."
            )

        loader = self.create_dataloader("train")
        raw_data = next(iter(loader))
        splitted_data = utils.split_and_load(raw_data, self.ctx)
        for data in splitted_data:
            inputs, labels = self.parse_data(data, "train")
            self.model(*inputs)
        self.model.export(os.path.join(self.config.PARAM_DIR, "model"), 9999)

    def train(self, test=True):
        for _ in range(self.latest_state + 1, self.config.MAX_EPOCHS):
            self.train_epoch()
            if test:
                self.test_epoch()
            self.latest_state += 1

    def train_epoch(self):
        s = time.time()
        self._process_epoch("train")
        t = time.time()
        print("train 1 epoch in {}\n".format(utils.parse_time(t - s)))

        epoch = self.latest_state + 1
        if (
            not self.config.SAVE_EPOCH_FREQ
            or epoch % self.config.SAVE_EPOCH_FREQ == 0
            or epoch == self.config.MAX_EPOCHS
        ):
            self.model.save_parameters(
                os.path.join(
                    self.config.PARAM_DIR,
                    "{}.params-{:04d}".format(self.config.PARAM_PREFIX, epoch),
                )
            )

    def test_epoch(self):
        s = time.time()
        self._process_epoch("test")
        t = time.time()
        print("test 1 epoch in {}\n\n\n".format(utils.parse_time(t - s)))

    def _process_epoch(self, mode):
        color_code = esc_seq.GREEN if sys.platform != "win32" else ""
        end_color_code = esc_seq.END if sys.platform != "win32" else ""
        print(
            color_code
            + "{}: epoch {:3d}/{:3d}".format(
                mode, self.latest_state + 1, self.config.MAX_EPOCHS
            )
            + end_color_code
        )

        loader = self.create_dataloader(mode)
        handler = self.create_handler(mode=mode, num_batch=len(loader))

        for i, raw_data in enumerate(loader):
            gathered_outputs = []
            gathered_losses = []
            losses = []
            tick = time.time()
            splitted_data = utils.split_and_load(raw_data, self.ctx)

            if mode == "train":
                autograd.set_training(True)
                autograd.set_recording(True)
            elif mode == "test":
                autograd.set_training(False)
                autograd.set_recording(False)

            for data in splitted_data:
                inputs, labels = self.parse_data(data, mode)
                outputs = self.parse_output(self.model(*inputs), mode)
                gathered_outputs.append(outputs)
                loss = self.compute_loss(outputs, labels)
                gathered_losses.append(loss)
                if mode == "train":
                    losses.extend(loss)

            autograd.set_training(False)
            autograd.set_recording(False)

            if mode == "train":
                autograd.backward(losses)
                self.trainer.step(raw_data[0].shape[0])

            handler.cleanup_batch(
                raw_data, gathered_outputs, gathered_losses, i, tick
            )

        handler.cleanup_epoch()

    def create_trainer(self):
        train_params = utils.collect_train_params(self.model, self.config)
        tmp_loader = self.create_dataloader("train")
        scheduler = self.create_lr_scheduler(len(tmp_loader))
        optimizer_params = self.config.OPT_PARAMS.copy()
        if isinstance(scheduler, lr_scheduler.LRScheduler):
            optimizer_params.update({"lr_scheduler": scheduler})
        return trainer.Trainer(
            train_params, self.config.OPTIMIZER, optimizer_params
        )

    def create_lr_scheduler(self, num_batch):
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
            shuffle=(mode == "train"),
            batchify_fn=dataset.batchify_fn,
            num_workers=self.config.NUM_WORKERS,
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
        return [data[0]], [data[1]]

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
        return (
            [outputs]
            if isinstance(outputs, nd.NDArray)
            else outputs
        )

    def backward_update(self, raw_data, outputs, losses):
        """Do backward.
        Usually it's enough to just backward the losses with autograd
        and do `optimizer.step(bs)`. But sometimes more customizations
        are needed, for example when only parts of the samples participate
        in training and thus `bs` should be customized.

        Parameters
        ----------
        raw_data: list of mxnet.nd.NDArray
            Data from dataloader.
        outputs: list of list of mxnet.nd.NDArray
            [[o0_gpu0, o1_gpu0, ...], ...].
        losses: list of mxnet.nd.NDArray
            [l0_gpu0, l1_gpu0, ...].
        """
        autograd.backward(losses)
        bs = raw_data[0].shape[0]
        self.trainer.step(bs)

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
        Specifically, `batch` is the current batch index and `tick` is the start
        time for processing the current batch.

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

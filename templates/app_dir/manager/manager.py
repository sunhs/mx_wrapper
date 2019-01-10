from mx_wrapper import manager


class Manager(manager.Manager):
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging."""
    def compute_loss(self, outputs, labels):
        pass

    def setup_handler(self, epoch, mode, num_batch, ds_size):
        pass

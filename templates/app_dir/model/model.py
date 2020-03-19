import mxnet as mx
from mxnet import initializer
from mxnet.gluon import nn


class Model(nn.HybridBlock):
    def __init__(self, config):
        """
        Initializes the model instance and calls the `initialize` method.

        :param config: The global config.
        """
        super(Model, self).__init__()
        self.config = config
        self.ctx = [mx.cpu(0)]
        if config.GPUS and mx.test_utils.list_gpus():
            self.ctx = [mx.gpu(i) for i in self.ctx]

    def initialize(
        self,
        init=initializer.Uniform(),
        ctx=None,
        verbose=False,
        force_reinit=False,
    ):
        super(Model, self).initialize(
            init=initializer.Xavier(rnd_type="uniform"),
            ctx=ctx,
            verbose=verbose,
            force_reinit=force_reinit,
        )

    def hybrid_forward(self, F, x, *args):
        # if getattr(self.config, "OUTPUT_PROB", None):
        #     outputs = F.softmax(outputs)
        pass

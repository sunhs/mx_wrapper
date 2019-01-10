import pickle

from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision import transforms

TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ]
)


class Dataset(dataset.Dataset):
    def __init__(self, config, mode):
        super(Dataset, self).__init__()
        self.config = config
        self.mode = mode
        with open(config.IMDB_PATH, 'rb') as f:
            self.imdb = pickle.load(f)[mode]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def batchify_fn(data):
        """Define this function if the default collate function defined in
        `mxnet.gluon.data.dataloader` doesn't meet your needs. Refer to the
        default one to find out how it works.

        Parameters
        ----------
        data: list
            List of data for one sample.

        Returns
        -------
        list
            Collated data.
        """
        raise NotImplementedError

import email.mime.image
import email.mime.multipart
import email.mime.text
import smtplib
import os

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn, parameter
import numpy as np


def parse_time(duration):
    hours = duration / 3600
    minutes = duration % 3600 / 60
    seconds = duration % 3600 % 60
    return '%dh:%dmin:%ds' % (hours, minutes, seconds)


def check_latest_param(param_dir, prefix):
    max_index = -1

    if not os.path.exists(param_dir) or not os.path.isdir(param_dir):
        return max_index

    if not prefix:
        raise ValueError("Specified param_dir but no prefix!")

    f_names = os.listdir(param_dir)
    for f_name in f_names:
        if not f_name.startswith(prefix):
            continue
        epoch = int(f_name.split('-')[1])
        if epoch > max_index:
            max_index = epoch
    return max_index


def init_model(
    model, param_dir='', prefix='', index=None, pretrain_path='', ctx=None
):
    latest_index = check_latest_param(param_dir, prefix)

    if latest_index != -1:
        if index is not None:
            assert index >= 0 and index <= latest_index
            latest_index = index
        load_path = os.path.join(
            param_dir, '{}.params-{:04d}'.format(prefix, latest_index)
        )
        print('==========>> resume from {}'.format(load_path))
        model.load_parameters(load_path, ctx)

    elif pretrain_path and os.path.exists(pretrain_path):
        print('==========>> loading from pretrain: {}'.format(pretrain_path))
        model.load_parameters(
            pretrain_path, ctx, allow_missing=True, ignore_extra=True
        )

    else:
        print('==========>> build from scratch')

    model.initialize(ctx=ctx)
    print('==========>> done')
    return latest_index


class MAP:
    def __init__(self):
        self.scores = None
        self.labels = None

    def add(self, scores, labels, copy=False):
        """Scores and target are both mxnet NDArray"""
        scores = scores.asnumpy()
        labels = labels.asnumpy()
        if self.scores is None or self.labels is None:
            if not copy:
                self.scores = scores
                self.labels = labels
            else:
                self.scores = scores.copy()
                self.labels = labels.copy()
            return

        self.scores = np.concatenate([self.scores, scores])
        self.labels = np.concatenate([self.labels, labels])

    def map(self):
        # copied from https://github.com/zxwu/lsvc2017
        probs = self.scores
        labels = self.labels
        mAP = np.zeros((probs.shape[1], ))

        for i in range(probs.shape[1]):
            iClass = probs[:, i]
            iY = labels[:, i]
            idx = np.argsort(-iClass)
            iY = iY[idx]
            count = 0
            ap = 0.0
            skip_count = 0
            for j in range(iY.shape[0]):
                if iY[j] == 1:
                    count += 1
                    ap += count / float(j + 1 - skip_count)
                if iY[j] == -1:
                    skip_count += 1
                if count != 0:
                    mAP[i] = ap / count
        return np.mean(mAP)
        # return mAP


def get_subblock_params(model, subblock_str):
    """[DEPRECATED] Originally used by `config_params`, which is now deprecated.
    Left here for backward compatibility.
    """
    children = subblock_str.split('.')
    block = model

    for child in children:
        block = block._children.get(child)

    return block.collect_params()


def set_lr_wd_mult(params, param_group):
    lr_mult = param_group.get('lr_mult', None)
    wd_mult = param_group.get('wd_mult', None)

    if lr_mult == 0:
        for p in params.values():
            p.grad_req = 'null'
        return False

    for p in params.values():
        if lr_mult is not None:
            p.lr_mult = lr_mult
        if wd_mult is not None:
            p.wd_mult = wd_mult

    return True


def config_params(model, config):
    """[DEPRECATED] Deprecated in favor of `collect_train_params`.
    Left here for backward compatibility.
    """
    all_params = model.collect_params()
    train_params = parameter.ParameterDict(all_params.prefix)
    non_default_param_keys = set()
    default_param_group = None

    for cfg_param_group in config.PARAM_GROUPS:
        if cfg_param_group['params'][0] == 'default':
            default_param_group = cfg_param_group
            continue

        grp_block_names = cfg_param_group['params']
        grp_params = parameter.ParameterDict(all_params.prefix)
        grp_param_keys = set()
        for block_name in grp_block_names:
            block_params = get_subblock_params(model, block_name)
            grp_params.update(block_params)
            grp_param_keys.update(set(block_params.keys()))
        non_default_param_keys.update(grp_param_keys)
        need_train = set_lr_wd_mult(grp_params, cfg_param_group)
        if need_train:
            train_params.update(grp_params)

    default_param_keys = set(all_params.keys()
                            ).difference_update(non_default_param_keys)

    if default_param_group and default_param_keys:
        default_params = parameter.ParameterDict(all_params.prefix)
        for key in default_param_keys:
            default_params.update({key: all_params[key]})
        need_train = set_lr_wd_mult(default_params, default_param_group)
        if need_train:
            train_params.update(default_params)

    return train_params


def collect_train_params(model, config):
    all_params = model.collect_params()
    train_params = parameter.ParameterDict(all_params.prefix)
    non_default_param_keys = set()
    default_param_group = None

    for cfg_param_group in config.PARAM_GROUPS:
        patterns = cfg_param_group['params']
        if isinstance(patterns, list):
            patterns = '|'.join(patterns)
        assert isinstance(patterns, str)

        if patterns == '.*':
            default_param_group = cfg_param_group
            continue

        grp_params = model.collect_params(patterns)
        grp_param_keys = set(grp_params.keys())
        non_default_param_keys.update(grp_param_keys)
        need_train = set_lr_wd_mult(grp_params, cfg_param_group)
        if need_train:
            train_params.update(grp_params)

    default_param_keys = set(all_params.keys()
                            ).difference_update(non_default_param_keys)

    if default_param_group and default_param_keys:
        default_params = parameter.ParameterDict(all_params.prefix)
        for key in default_param_keys:
            default_params.update({key: all_params[key]})
        need_train = set_lr_wd_mult(default_params, default_param_group)
        if need_train:
            train_params.update(default_params)

    # Ignore default params while training.
    if default_param_keys:
        for key in default_param_keys:
            all_params[key].grad_req = 'null'

    return train_params


def split_and_load(data, ctx=[mx.cpu()]):
    splitted_data = []
    for _data in data:
        if isinstance(_data, nd.NDArray):
            splitted_data.append(
                gluon.utils.split_and_load(_data, ctx, even_split=False)
            )
        elif isinstance(_data, (tuple, list)):
            assert len(_data) % len(ctx) == 0, \
                'Batch size should be divisible by ctx count.'
            chunk_size = len(_data) / len(ctx)
            chunks = []
            for i in range(chunk_size):
                chunks.append(_data[i * chunk_size : (i + 1) * chunk_size])
            splitted_data.append(chunks)
        else:
            raise TypeError(
                'Data should be either mxnet NDArray, tuple or list'
            )
    return list(zip(*splitted_data))


def send_email(sender_info, receiver, subject, content, images=None):
    """Send email.

    Example content with images:
    'This is an example.
     <img src="cid:image1">'
    In this example, you provide the cid `image1` for your image. So you should
    provide it and the image path for the `images` parameter.

    Args:
      sender_info (dict): {'email':    *sender email address*,
                           'password': *sender email password*,
                           'server':   *sender smtp server* (i.e. smtp.163.com)}
      receiver     (str): Receiver email address.
      subject      (str): Email subject.
      content      (str): Text or HTML content. If images are provided, each of
        them should be attached with a cid (in your HTML tag). This cid and the
        image file path should be provided for the parameter `images`.
      images      (list): Each element is a dict with keys `path`, `cid`.
    """
    msg_root = email.mime.multipart.MIMEMultipart('related')
    msg_root['Subject'] = subject
    msg_root['From'] = sender_info['email']
    msg_root['To'] = receiver

    msg_text = email.mime.text.MIMEText(content, 'html', 'utf-8')
    msg_root.attach(msg_text)

    if images is not None:
        for image in images:
            with open(image['path'], 'rb') as f:
                msg_image = email.mime.image.MIMEImage(f.read())
            msg_image.add_header('Content-ID', '<{}>'.format(image['cid']))
            msg_root.attach(msg_image)

    smtp = smtplib.SMTP()
    smtp.connect(sender_info['server'])
    smtp.login(sender_info['email'], sender_info['password'])
    smtp.sendmail(sender_info['email'], receiver, msg_root.as_string())
    smtp.quit()

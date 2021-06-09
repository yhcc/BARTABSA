import numpy as np


def get_max_len_max_len_a(data_bundle, max_len=10):
    """

    :param data_bundle:
    :param max_len:
    :return:
    """
    max_len_a = -1
    for name, ds in data_bundle.iter_datasets():
        if name=='train':continue
        src_seq_len = np.array(ds.get_field('src_seq_len').content)
        tgt_seq_len = np.array(ds.get_field('tgt_seq_len').content)
        _len_a = round(max(np.maximum(tgt_seq_len - max_len+2, 0)/src_seq_len), 1)

        if _len_a>max_len_a:
            max_len_a = _len_a

    return max_len, max_len_a


def get_num_parameters(model):
    num_param = 0
    for name, param in model.named_parameters():
        num_param += np.prod(param.size())
    print(f"The number of parameters is {num_param}")
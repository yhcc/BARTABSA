import numpy as np
import fitlog
# from .metrics import Seq2SeqSpanMetric
from collections import defaultdict
from fastNLP import Tester
from fastNLP import SortedSampler


def get_max_len_max_len_a(data_bundle, max_len=10):
    """
    当给定max_len=10的时候计算一个最佳的max_len_a

    :param data_bundle:
    :param max_len:
    :return:
    """
    max_len_a = -1
    for name, ds in data_bundle.iter_datasets():
        if name=='train':continue
        if ds.has_field('src_seq_len'):
            src_seq_len = np.array(ds.get_field('src_seq_len').content)
            tgt_seq_len = np.array(ds.get_field('tgt_seq_len').content)
            _len_a = round(max(np.maximum(tgt_seq_len - max_len+2, 0)/src_seq_len), 1)

            if _len_a>max_len_a:
                max_len_a = _len_a
    if not fitlog.is_debug():
        fitlog.add_hyper(name='max_len', value=max_len)
        fitlog.add_hyper(name='max_len_a', value=max_len_a)

    return max_len, max_len_a


def iterate_over_length_penalty_and_beam_size(model, data_bundle, label_ids, eos_token_id, batch_size,
                                              dataset_name, log_id, use_tqdm=True):
    """
    会iterater所有的beam size和length_penalty组合

    :param model:
    :param data_bundle:
    :return:
    """
    if fitlog.is_debug():
        return

    # num_beam为1只需要跑一次就好了
    model.generator.set_new_generator(num_beams=1)

    metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids))
    results = defaultdict(list)
    for name in ['dev', 'test']:
        if name not in data_bundle.datasets:
            continue
        results['length_penalty'].append('-')
        results['num_beams'].append(1)
        results['split_name'].append(name)
        if dataset_name is not None:
            results['dataset_name'].append(dataset_name)
        if log_id is not None:
            results['log_id'].append(log_id)
        ds = data_bundle.get_dataset(name)
        tester = Tester(data=ds, model=model, metrics=metric, batch_size=batch_size, num_workers=2, device=None, verbose=1,
                        use_tqdm=use_tqdm,
                        fp16=False, sampler=SortedSampler('src_seq_len'))
        eval_res = tester.test()
        metric_res = list(eval_res.values())[0]
        for key, value in metric_res.items():
            results[key].append(value)

    for length_penalty in [0.8, 1.0, 1.5, 2.0]:
        for num_beam in [2, 4, 6]:
            model.generator.set_new_generator(num_beams=num_beam, length_penalty=length_penalty)
            for name in ['dev', 'test']:
                if name not in data_bundle.datasets:
                    continue
                print(f'\nsplit:{name}, length_penalty:{length_penalty}, num_beams:{num_beam}')

                results['length_penalty'].append(length_penalty)
                results['num_beams'].append(num_beam)
                results['split_name'].append(name)
                if dataset_name is not None:
                    results['dataset_name'].append(dataset_name)
                if log_id is not None:
                    results['log_id'].append(log_id)
                ds = data_bundle.get_dataset(name)
                tester = Tester(data=ds, model=model, metrics=metric, batch_size=batch_size, num_workers=4, device=None, verbose=1,
                                use_tqdm=use_tqdm,
                                fp16=False, sampler=SortedSampler('src_seq_len'))
                try:
                    eval_res = tester.test()
                    metric_res = list(eval_res.values())[0]
                except:
                    max_len = max(map(len, results.values()))
                    metric_res = {}
                    for key in results.keys():
                        if len(results[key])<max_len:  # 说明需要补充
                            metric_res[key] = -1

                for key, value in metric_res.items():
                    results[key].append(value)

    assert len(set(map(len, results.values())))==1

    key_lengths = list(map(len, results.keys()))
    value_lengths = [max(list(map(len, list(map(str, values))))) for values in results.values()]
    sep = 3
    lengths = []
    for len1, len2 in zip(key_lengths, value_lengths):
        lengths.append(max(len1, len2)+sep)

    lines = []
    list_results = [list(results.keys())] + list(zip(*results.values()))
    formatter = ''
    for length in lengths:
        formatter += '{:>%i} ' % (length)
    for values in list_results:
        line = formatter.format(*values)
        lines.append(line)

    from fitlog import _logger
    if _logger._log_dir is not None:
        fitlog.add_to_line('\n'.join(lines))
    else:
        print('\n'.join(lines))
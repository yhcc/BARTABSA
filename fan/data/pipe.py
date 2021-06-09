from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key


def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class BartBPEABSAPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first=False):
        super(BartBPEABSAPipe, self).__init__()
        assert opinion_first is False
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping = {
            'tag': '<<tag>>',
        }
        self.opinion_first = opinion_first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'polarity': str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos, 然后是

        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.tokenizer.eos_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # 特殊的开始
            target_spans = []
            _word_bpes = list(chain(*word_bpes))

            # todo 这里需要解决一个当opinion first的时候，按照opinion排序，否则按照aspects排序
            aspects = ins['aspects']
            assert len(aspects) == 1
            aspects = aspects[0]
            opinions = ins['opinions']
            o_bpes = []
            a_start_bpe = cum_lens[aspects['from']]  # 因为有一个sos shift
            a_end_bpe = cum_lens[aspects['to'] - 1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
            for idx, word in zip((a_start_bpe, a_end_bpe),
                                 (aspects['term'][0], aspects['term'][-1])):
                assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                       _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

            for opinion in opinions:
                o_start_bpe = cum_lens[opinion['from']]  # 因为有一个sos shift
                o_end_bpe = cum_lens[opinion['to']-1]  # 因为有一个sos shift
                o_bpes.extend([o_start_bpe+target_shift, o_end_bpe+target_shift, target_shift-1])

                # 这里需要evaluate是否是对齐的
                for idx, word in zip((o_start_bpe, o_end_bpe),
                                     (opinion['term'][0], opinion['term'][-1])):
                    assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                           _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

                target_spans.append((o_start_bpe+target_shift, o_end_bpe+target_shift))
            target.extend([a_start_bpe+target_shift, a_end_bpe+target_shift] + o_bpes)
            target.append(1)  # append 1是由于特殊的eos

            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes))}

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = FanABSALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class FanABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # TODO 这里首先需要把数据merge到一起，相同的input的
        new_data = {}
        merge_line = 0
        # 如果是train的话，需要把aspect一样的opinion和在一起
        if 'train' in path:
            for d in data:
                assert len(d['aspects'])==1
                key = (d['raw_words'], d['aspects'][0]['from'], d['aspects'][0]['to'])
                if key in new_data:
                    merge_line += 1
                    new_data[key]['opinions'].extend(d['opinions'])
                else:
                    new_data[key] = d
            new_data = new_data.values()
        else:
            new_data = data
        ds = DataSet()
        for ins in new_data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            # assert len(aspects)==len(opinions)
            ins = Instance(raw_words=tokens, aspects=aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        print(f"Merge {merge_line} lines from old:{len(data)} to new:{len(new_data)}.")
        return ds


if __name__ == '__main__':
    data_bundle = BartBPEABSAPipe().process_from_file('../../../data/pengb/14lap')
    print(data_bundle)


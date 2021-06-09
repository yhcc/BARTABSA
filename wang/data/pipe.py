from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key


def cmp(v1, v2):
    if v1['from']==v2['from']:
        return v1['to'] - v2['to']
    return v1['from'] - v2['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class WangBartABSAPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first=True):
        super(WangBartABSAPipe, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping = {  # 应该是要在他们的前面，
            "OE": "<<opinion_extraction>>",
            "AESC": "<<aspect_extraction>>",  # 加是为了保证排序在前面
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>',
            "CON": '<<conflict>>'
        }
        self.opinion_first = opinion_first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_tokens = cur_num_tokens

    def add_tokens(self):
        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        # 需要保证AESC和OE是前两位的
        for i, value in enumerate(["<<opinion_extraction>>", '<<aspect_extraction>>']):
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))[0]
            assert key_id==self.cur_num_tokens+i

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)+2

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
        self.add_tokens()
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos

        # todo 需要融入task id
        for name in ['train', 'dev', 'test']:
            ds = data_bundle.get_dataset(name)
            o_ds = DataSet()  # 用来做opinion的
            if name == 'train':
                a_ds = o_ds  # 用来做aspect的
            else:
                a_ds = DataSet()
            for ins in ds:
                raw_words = ins['raw_words']
                word_bpes = [[self.tokenizer.bos_token_id]]
                for word in raw_words:
                    bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                    bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                    word_bpes.append(bpes)
                word_bpes.append([self.tokenizer.eos_token_id])

                lens = list(map(len, word_bpes))
                cum_lens = np.cumsum(list(lens)).tolist()
                o_target = [0, self.mapping2targetid['OE'], self.mapping2targetid['OE']]  # 特殊的开始
                a_target = [0, self.mapping2targetid['AESC'], self.mapping2targetid['AESC']]
                aesc_target_spans = []
                ae_target_spans = []
                oe_target_spans = []
                _word_bpes = list(chain(*word_bpes))

                aspects = sorted(ins['aspects'], key=cmp_to_key(cmp))
                opinions = sorted(ins['opinions'], key=cmp_to_key(cmp))

                for aspect in aspects:
                    s_bpe = cum_lens[aspect['from']]+target_shift
                    e_bpe = cum_lens[aspect['to']-1]+target_shift
                    polarity = self.mapping2targetid[aspect['polarity']]
                    ae_target_spans.append((s_bpe, e_bpe))
                    a_target.extend([s_bpe, e_bpe, polarity])
                    # sc_target_spans.append(polarity)
                    aesc_target_spans.append((s_bpe, e_bpe, polarity))

                for opinion in opinions:
                    s_bpe = cum_lens[opinion['from']]+target_shift
                    e_bpe = cum_lens[opinion['to']-1]+target_shift
                    oe_target_spans.append((s_bpe, e_bpe))
                    o_target.extend((s_bpe, e_bpe))

                a_target.append(1)
                o_target.append(1)
                o_ins = Instance(src_tokens=_word_bpes.copy(), tgt_tokens=o_target)
                a_ins = Instance(src_tokens=_word_bpes.copy(), tgt_tokens=a_target)
                if name!='train':
                    a_ins.add_field('ae_target_span', ae_target_spans)
                    # a_ins.add_field('sc_target_span', sc_target_spans)
                    a_ins.add_field('aesc_target_span', aesc_target_spans)
                    o_ins.add_field('oe_target_span', oe_target_spans)
                o_ds.append(o_ins)
                a_ds.append(a_ins)

            if name == 'train':
                data_bundle.set_dataset(a_ds, 'train')
            else:
                data_bundle.set_dataset(a_ds, name+'a')
                data_bundle.set_dataset(o_ds, name+'o')

        data_bundle.set_ignore_type('oe_target_span', "aesc_target_span", "ae_target_span")
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'oe_target_span', 'aesc_target_span',
                                'ae_target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        # if '15res' in paths:
        #     self.mapping.pop('CON')
        data_bundle = ABSALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class ABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        delete = 0
        for ins in data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            if len(opinions)==1:
                if len(opinions[0]['term'])==0:
                    opinions = []
            if len(aspects)==1:
                if len(aspects[0]['term'])==0:
                    aspects = []
            new_aspects = []
            for aspect in aspects:
                if 'polarity' not in aspect:
                    delete += 1
                    continue
                new_aspects.append(aspect)

            ins = Instance(raw_words=tokens, aspects=new_aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        print(f"For path:{path}, delete {delete} conflicts.")
        return ds


if __name__ == '__main__':
    data_bundle = WangBartABSAPipe().process_from_file('../../../data/pengb/14lap')
    print(data_bundle)


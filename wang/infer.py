
import sys
sys.path.append('../')
from transformers import AutoTokenizer
import torch
from fastNLP import cache_results
import numpy as np
from itertools import chain
from copy import deepcopy


# 这一段代码的目的是为了加载cache好的东西，因为里面包含了tokenizer，mappping2id什么的，如果是单独额外保存的话，可以用其他方式加载的
bart_name = 'facebook/bart-base'
dataset_name = 'wang/15res'
opinion_first = False
cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}.pt"
@cache_results(cache_fn, _refresh=False)
def get_data():
    # 因为实际上是加载，不会load
    pass

_, tokenizer, mapping2id, mapping2targetid = get_data()
# target decode这边的prompt
o_target = torch.LongTensor([0, mapping2targetid['OE'], mapping2targetid['OE']])  # 特殊的开始
a_target = torch.LongTensor([0, mapping2targetid['AESC'], mapping2targetid['AESC']])
eos_token_id = 1
target_shift = len(mapping2id)+2
labels = list(mapping2id.keys())
word_start_index = len(labels)+2
# 加载model
model = torch.load('save_models/best_SequenceGeneratorModel_aesc_f_2022-06-23-13-43-58-555586')
# model.generator.set_new_generator() # 可以重新设置一些与生成相关的参数


# 准备数据
sents = [
    'judging from previous posts this used to be a good place , but not any longer',
    'we , there were four of us , arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude'
]

# 数据准备
src_tokens = []
src_seq_len = []
mappingbacks = []
aesc_tgt_tokens = []
oe_tgt_tokens = []
raw_words = []
for sent in sents:
    _raw_words = sent.split()
    raw_words.append(_raw_words)
    word_bpes = [[tokenizer.bos_token_id]]
    for word in _raw_words:
        bpes = tokenizer.tokenize(word, add_prefix_space=True)
        bpes = tokenizer.convert_tokens_to_ids(bpes)
        word_bpes.append(bpes)
    word_bpes.append([tokenizer.eos_token_id])
    lens = list(map(len, word_bpes))
    cum_lens = np.cumsum(list(lens)).tolist()
    mappingback = np.full(shape=(cum_lens[-1]+1), fill_value=-1, dtype=int).tolist()
    for _i, _j in enumerate(cum_lens):
        mappingback[_j] = _i
    mappingbacks.append(mappingback)
    src_tokens.append(torch.LongTensor(list(chain(*word_bpes))))
    src_seq_len.append(len(src_tokens[-1]))
    aesc_tgt_tokens.append(a_target)
    oe_tgt_tokens.append(o_target)


encoder_inputs = {'src_tokens': torch.nn.utils.rnn.pad_sequence(src_tokens, batch_first=True,
                                                                padding_value=tokenizer.pad_token_id),
                 'src_seq_len': torch.LongTensor(src_seq_len)}

# 因为是共享encode的，所以单独拆分成encode和generate
model.eval()
with torch.no_grad():
    state = model.seq2seq_model.prepare_state(**encoder_inputs)
    print(state.num_samples)
    aesc_result = model.generator.generate(deepcopy(state), tokens=torch.stack(aesc_tgt_tokens, dim=0))  # the prompt is provided to the model
    oe_result = model.generator.generate(deepcopy(state), tokens=torch.stack(oe_tgt_tokens, dim=0))

# 抽取aesc数据
aspects = []
aesc_eos_index = aesc_result.flip(dims=[1]).eq(eos_token_id).cumsum(dim=1).long()
aesc_result = aesc_result[:, 1:]  # delete </s>
aesc_seq_len = aesc_eos_index.flip(dims=[1]).eq(aesc_eos_index[:, -1:]).sum(dim=1)  # bsz
aesc_seq_len = (aesc_seq_len - 2).tolist()
for i, (ps, length, mappingback, _raw_words) in enumerate(zip(aesc_result.tolist(), aesc_seq_len,
                                                              mappingbacks, raw_words)):
    ps = ps[2:length]
    pairs = []
    cur_pair = []  # each pair with the format (a_start, a_end, class), start/end inclusive, and considering the sos in the start of sentence
    if len(ps):
        for index, j in enumerate(ps):
            if j < word_start_index:
                cur_pair.append(j)
                if len(cur_pair) != 3 or cur_pair[0] > cur_pair[1]:
                    pass
                else:
                    pairs.append(tuple(cur_pair))
                cur_pair = []
            else:
                cur_pair.append(j)
    _aspects = []
    for index, pair in enumerate(pairs):
        a_s, a_e, sc = pair
        a_s, a_e = mappingback[a_s-target_shift], mappingback[a_e-target_shift]+1
        _aspects.append({
            'index': index,
            'from': a_s,
            'to': a_e,
            'polarity': labels[sc-2].upper(),
            'term': ' '.join(_raw_words[a_s:a_e])
        })
    aspects.append(_aspects)

# 抽取oe数据
opinons = []
oe_eos_index = oe_result.flip(dims=[1]).eq(eos_token_id).cumsum(dim=1).long()
oe_result = oe_result[:, 1:]  # delete </s>
oe_seq_len = oe_eos_index.flip(dims=[1]).eq(oe_eos_index[:, -1:]).sum(dim=1)  # bsz
oe_seq_len = (oe_seq_len - 2).tolist()
for i, (ps, length, mappingback, _raw_words) in enumerate(zip(oe_result.tolist(), oe_seq_len,
                                                              mappingbacks, raw_words)):
    ps = ps[2:length]  # 去掉prompt
    print(ps)
    pairs = []
    cur_pair = []  # each pair with the format (a_start, a_end, class), start/end inclusive, and considering the sos in the start of sentence
    if len(ps):
        for index, j in enumerate(ps, start=1):
            if index%2==0:
                cur_pair.append(j)
                if cur_pair[0]>cur_pair[1] or cur_pair[0]<word_start_index\
                        or cur_pair[1]<word_start_index:
                    invalid = 1
                else:
                    pairs.append(tuple(cur_pair))
                cur_pair = []
            else:
                cur_pair.append(j)
    _opinons = []
    for index, pair in enumerate(pairs):
        o_s, o_e = pair
        o_s, o_e = mappingback[o_s-target_shift], mappingback[o_e-target_shift]+1
        _opinons.append({
            'index': index,
            'from': o_s,
            'to': o_e,
            'term': ' '.join(_raw_words[o_s:o_e])
        })
    opinons.append(_opinons)


print(aspects)

print(opinons)




import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from data.pipe import BartBPEABSAPipe
from peng.model.bart_absa import BartSeq2SeqModel

from fastNLP import Trainer
from peng.model.metrics import Seq2SeqSpanMetric
from peng.model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from peng.model.generator import SequenceGeneratorModel
import fitlog

fitlog.debug()
fitlog.set_log_dir('logs')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='pengb/14lap', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--save_model', type=int, default=0)

args= parser.parse_args()

lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp
save_model = args.save_model
fitlog.add_hyper(args)

#######hyper
#######hyper


demo = False
if demo:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}_demo.pt"
else:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}.pt"


@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()
max_len = 10
max_len_a = {
    'penga/14lap': 0.9,
    'penga/14res': 1,
    'penga/15res': 1.2,
    'penga/16res': 0.9,
    'pengb/14lap': 1.1,
    'pengb/14res': 1.2,
    'pengb/15res': 0.9,
    'pengb/16res': 1.2
}[dataset_name]

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None)

import torch
if torch.cuda.is_available():
    # device = list([i for i in range(torch.cuda.device_count())])
    device = 'cuda'
else:
    device = 'cpu'

parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':0}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)


callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))

sampler = None
# sampler = ConstTokenNumSampler('src_seq_len', max_token=1000)
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)


model_path = None
if save_model:
    model_path = 'save_models/'

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=2, n_epochs=n_epochs, print_every=1,
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key='triple_f',
                  validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                  callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size)

trainer.train(load_best_model=True)

from fastNLP import MetricBase, Tester
import numpy as np
test = data_bundle.get_dataset('test')
test.set_ignore_type('raw_words', 'aspects', 'opinions')
test.set_target('raw_words', 'aspects', 'opinions')
class WriteResultToFileMetric(MetricBase):
    def __init__(self, target_shift, labels, fp, tokenizer, eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first):
        super(WriteResultToFileMetric, self).__init__()
        self.fp = fp
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos
        self.opinin_first = opinion_first
        self.tokenizer = tokenizer
        self.target_shift = target_shift
        self.labels = labels

        self.raw_words = []
        self.aspect_preds = []
        self.opinion_preds = []
        self.aspect_targets = []
        self.opinion_targets = []

    def evaluate(self, target_span, raw_words, aspects, opinions, pred):
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        for i, (_opinions, _raw_words, ts, ps) in enumerate(zip(opinions, raw_words, target_span, pred.tolist())):
            # use ts to check whether the decoding is right or not
            word_bpes = [[self.tokenizer.bos_token_id]]
            for word in _raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.tokenizer.eos_token_id])
            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            mappingback = np.full(shape=(cum_lens[-1]+1), fill_value=-1, dtype=int).tolist()
            for _i, _j in enumerate(cum_lens):
                mappingback[_j] = _i

            _opinion_start_indexes = set([o['from'] for o in _opinions])
            _opinion_end_indexes = set([o['to'] for o in _opinions])
            _labels = set([a['polarity'] for a in aspects[i]])

            for _t in ts:
                o_s, o_e = mappingback[_t[2]-self.target_shift], mappingback[_t[3]-self.target_shift]+1
                assert o_s!=-1 or o_e!=-1, (o_s, o_e)
                assert o_s in _opinion_start_indexes, o_e in _opinion_end_indexes
                assert self.labels[_t[4]-2] in _labels

            ps = ps[:pred_seq_len[i]]
            pairs = []
            cur_pair = []  # each pair with the format (a_start, a_end, o_start, o_end, class), start/end inclusive, and considering the sos in the start of sentence
            if len(ps):
                for index, j in enumerate(ps):
                    if j < self.word_start_index:
                        cur_pair.append(j)
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                            pass
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            _aspects = []
            _opinions = []
            for index, pair in enumerate(pairs):
                a_s, a_e, o_s, o_e, sc = pair
                o_s, o_e, a_s, a_e = mappingback[o_s-self.target_shift], mappingback[o_e-self.target_shift]+1, \
                                     mappingback[a_s-self.target_shift], mappingback[a_e-self.target_shift]+1
                _aspects.append({
                    'index': index,
                    'from': a_s,
                    'to': a_e,
                    'polarity': self.labels[sc-2].upper(),
                    'term': ' '.join(_raw_words[a_s:a_e])
                })
                _opinions.append({
                    'index': index,
                    'from': o_s,
                    'to': o_e,
                    'term': ' '.join(_raw_words[o_s:o_e])
                })
            self.aspect_preds.append(_aspects)
            self.opinion_preds.append(_opinions)

        self.raw_words.extend(raw_words.tolist())
        self.aspect_targets.extend(aspects.tolist())
        self.opinion_targets.extend(opinions.tolist())

    def get_metric(self, reset=True):
        data = []
        for raw_words, aspect_targets, opinion_targets, aspect_preds, opinion_preds in zip(
          self.raw_words, self.aspect_targets, self.opinion_targets, self.aspect_preds, self.opinion_preds
        ):
            data.append({
                'words': raw_words,
                'aspects': aspect_targets,
                'opinions': opinion_targets,
                'aspect_preds': aspect_preds,
                'opinion_preds': opinion_preds
            })
        import json
        line = json.dumps(data, indent=1)
        with open(self.fp, 'w', encoding='utf-8') as f:
            f.write(line)
        return {}

fp = os.path.split(args.dataset_name)[-1] + '.txt'
tester = Tester(test, model, metrics=[metric,
                                      WriteResultToFileMetric(len(mapping2id)+2, list(mapping2id.keys()), fp, tokenizer, eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)],
                batch_size=64, device=0)
tester.test()


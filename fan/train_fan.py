import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from fan.data.pipe import BartBPEABSAPipe
from fan.model.bart_fan import BartSeq2SeqModel, Restricter

from fastNLP import Trainer, CrossEntropyLoss, Tester
from fan.model.metrics import Seq2SeqSpanMetric
from fan.model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fan.model.callbacks import FitlogCallback
from fastNLP.core.sampler import  ConstTokenNumSampler
import fitlog
from fastNLP.core.sampler import SortedSampler
from fan.model.generater import SequenceGeneratorModel
from fan.model.utils import get_max_len_max_len_a
# fitlog.debug()
# fitlog.commit(__file__)
fitlog.set_log_dir('logs')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='fan/14lap', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score', 'avg_feature'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)

# parser.add_argument('--use_encoder_mlp', action='store_true', default=False)


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
fitlog.add_hyper(args)
use_encoder_mlp = args.use_encoder_mlp


#######hyper
#######hyper


demo = False
if demo:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}_demo.pt"
else:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}.pt"

# 这里生成的数据，是没有first而是直接bpe的
@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()
max_len = 10
max_len_a = {
    'fan/14lap': 0.5,
    'fan/14res': 0.7,
    'fan/15res': 0.4,
    'fan/16res': 0.6
}[dataset_name]

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  # 因为是特殊符号
eos_token_id = 1  # 因为是特殊符号
label_ids = list(mapping2id.values())
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)

vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
restricter = Restricter(label_ids)
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None)

import torch
if torch.cuda.is_available():
    # device = list([i for i in range(torch.cuda.device_count())])
    if 'p' not in os.environ and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        device = 'cuda'
    else:
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
if 'dev' in data_bundle.datasets:
    callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))
    dev_data = data_bundle.get_dataset('dev')
else:
    callbacks.append(FitlogCallback())
    dev_data = data_bundle.get_dataset('test')


sampler = None
# sampler = ConstTokenNumSampler('src_seq_len', max_token=1000)
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)


trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=2, n_epochs=n_epochs, print_every=1,
                  dev_data=dev_data, metrics=metric, metric_key='oe_f',
                  validate_every=-1, save_path=None, use_tqdm='SEARCH_ID' not in os.environ, device=device,
                  callbacks=callbacks, check_code_level=-1 if 'SEARCH_ID' in os.environ else 0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size*5)

trainer.train(load_best_model=False)

if trainer.save_path is not None:
    model_name = "best_" + "_".join([model.__class__.__name__, trainer.metric_key, trainer.start_time])
    fitlog.add_other(name='model_name', value=model_name)
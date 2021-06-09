import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from wang.data.pipe import WangBartABSAPipe
from wang.model.bart_wang import BartSeq2SeqModel, Restricter

from fastNLP import Trainer, CrossEntropyLoss, Tester
from wang.model.metrics import OESpanMetric, AESCSpanMetric
from wang.model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results
from wang.model.callbacks import FitlogCallback, WarmupCallback
import fitlog
from fastNLP.core.sampler import SortedSampler
from wang.model.generater import SequenceGeneratorModel
# fitlog.debug()
# fitlog.commit(__file__)
fitlog.set_log_dir('logs')
fitlog.add_hyper_in_file(__file__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='wang/15res', type=str)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=30, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score', 'avg_feature'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--warmup', type=float, default=0.1)
# parser.add_argument('--use_encoder_mlp', action='store_true', default=False)


args= parser.parse_args()

if args.dataset_name == 'wang/15res':
    args.decoder_type = 'avg_score'
    args.n_epochs = 35
    args.use_encoder_decoder = 1
    args.warmup = 0.01
elif args.dataset_name == 'wang/14lap':
    args.decoder_type = 'avg_score'
    args.n_epochs = 30
    args.use_encoder_decoder = 1
    args.warmup = 0.01
    args.lr = 4e-5
elif args.dataset_name == 'wang/14res':
    args.decoder_type = 'avg_feature'
    args.n_epochs = 20
    args.use_encoder_decoder = 1
    args.warmup = 0.1
    args.lr =3e-5

lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
warmup = args.warmup
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
    pipe = WangBartABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid

data_bundle, tokenizer, mapping2id, mapping2targetid = get_data()
conflict_id = -1 if 'CON' not in mapping2targetid else mapping2targetid['CON']
print(data_bundle)
max_len = 10
max_len_a = {
    'wang/14lap': 0.6,
    'wang/14res': 0.6,
    'wang/15res': 0.5
}[dataset_name]

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  # 因为是特殊符号
eos_token_id = 1  # 因为是特殊符号 TODO 特别需要注意这是1, 因为把这些token都做了重映射
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
        device = 'cuda:1'
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
callbacks.append(WarmupCallback(warmup=warmup, schedule='linear'))
callbacks.append(FitlogCallback(tester={
    'testa': Tester(data=data_bundle.get_dataset('testa'), model=model,
                    metrics=AESCSpanMetric(eos_token_id, num_labels=len(label_ids), conflict_id=conflict_id),
                    batch_size=batch_size*5, num_workers=2, device=None, verbose=0, use_tqdm=False,
                    fp16=False),
    'testo': Tester(data=data_bundle.get_dataset('testo'), model=model,
                    metrics=OESpanMetric(eos_token_id, num_labels=len(label_ids)),
                    batch_size=batch_size*5, num_workers=2, device=None, verbose=0, use_tqdm=False,
                    fp16=False),
    # 'devo': Tester(data=data_bundle.get_dataset('devo'), model=model,
    #               metrics=OESpanMetric(eos_token_id, num_labels=len(label_ids)),
    #               batch_size=batch_size*5, num_workers=2, device=None, verbose=0, use_tqdm=False,
    #               fp16=False)
}))

dev_data = data_bundle.get_dataset('deva')

sampler = None
# sampler = ConstTokenNumSampler('src_seq_len', max_token=3000)
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = [AESCSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=False, conflict_id=conflict_id)]
# data_bundle.get_dataset('train').drop(lambda ins:ins['tgt_tokens'][1]==3, inplace=True)

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=2, n_epochs=n_epochs, print_every=1 if 'SEARCH_OUTPUT_FP' not in os.environ else 100,
                  dev_data=dev_data, metrics=metric, metric_key='aesc_f',
                  validate_every=-1, save_path=None, use_tqdm='SEARCH_ID' not in os.environ, device=device,
                  callbacks=callbacks, check_code_level=-1 if 'SEARCH_ID' in os.environ else 0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size*5)

trainer.train(load_best_model=False)

if trainer.save_path is not None:
    model_name = "best_" + "_".join([model.__class__.__name__, trainer.metric_key, trainer.start_time])
    fitlog.add_other(name='model_name', value=model_name)
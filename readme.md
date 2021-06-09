This is the code for ACL2021 paper [A Unified Generative Framework for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2106.04300)

Install the package in the requirements.txt, then use the following
commands to install two other packages
```text
pip install git+https://github.com/fastnlp/fastNLP@dev
pip install git+https://github.com/fastnlp/fitlog
```

The structure of this code is as follows
```text
 -  data
    - fan  # D_19 in paper
    - penga  # D_20a in paper
    - pengb  # D_20b in paper
    - wang  # D_17 in paper
- fan/
    train_fan.py  # training file for fan data
- peng/
    train.py  # training file for penga and pengb
- wang/
    train_wang.py  # training file for wang
```
Please do remember to cite these dataset paper if you use them.

After enter the folder, you can run the code by directly using
```shell
python train.py --dataset pengb/14lap
```
The following output should be achieved
```text
Save cache to caches/data_facebook/bart-base_pengb/14lap_False.pt.                                                                   
The number of tokens in tokenizer  50265
50268 50273
The number of parameters is 140607744
input fields after batch(if batch size is 2):
        tgt_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22]) 
        src_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 41]) 
        src_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
        tgt_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
target fields after batch(if batch size is 2):
        tgt_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22]) 
        target_span: (1)type:numpy.ndarray (2)dtype:object, (3)shape:(2,) 
        tgt_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 

training epochs started 2021-02-03-02-24-46-454466
Evaluate data in 6.19 seconds!                                                                                                       
Evaluate data in 12.32 seconds!                                                                                                      
EvaluateCallback evaluation on data-test:                                                                                            
Seq2SeqSpanMetric: triple_f=23.86, triple_rec=16.45, triple_pre=43.41, oe_ae_f=27.35, oe_ae_rec=18.85, oe_ae_pre=49.76, ae_sc_f=33.28
, ae_sc_rec=23.97, ae_sc_pre=54.410000000000004, em=0.1494, invalid=0.436
Evaluation on dev at Epoch 1/50. Step:57/2850:                                                                                       
Seq2SeqSpanMetric: triple_f=21.47, triple_rec=14.78, triple_pre=39.23, oe_ae_f=24.42, oe_ae_rec=16.81, oe_ae_pre=44.62, ae_sc_f=33.80
0000000000004, ae_sc_rec=24.32, ae_sc_pre=55.379999999999995, em=0.1507, invalid=0.4384

....

In Epoch:50/Step:2850, got best dev performance:
Seq2SeqSpanMetric: triple_f=58.03, triple_rec=57.099999999999994, triple_pre=58.98, oe_ae_f=63.92, oe_ae_rec=62.9, oe_ae_pre=64.97, ae_sc_f=73.91, ae_sc_rec=74.66000000000001, ae_sc_pre=73.18, em=0.4155, invalid=0.0502
```


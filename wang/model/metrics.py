
from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter


class OESpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, opinion_first=True):
        super(OESpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.word_start_index = num_labels+2

        self.oe_fp = 0
        self.oe_tp = 0
        self.oe_fn = 0
        self.em = 0
        self.total = 0
        self.invalid = 0
        # assert opinion_first is False, "Current metric only supports aspect first"

        self.opinin_first = opinion_first

    def evaluate(self, oe_target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(oe_target_span, pred.tolist())):
            em = 0
            assert ps[0]==tgt_tokens[i, 0]
            ps = ps[2:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps, start=1):
                    if index%2==0:
                        cur_pair.append(j)
                        if cur_pair[0]>cur_pair[1] or cur_pair[0]<self.word_start_index\
                                or cur_pair[1]<self.word_start_index:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            self.invalid += invalid

            oe_target_counter = Counter([tuple(t) for t in ts])
            oe_pred_counter = Counter(pairs)

            # 这里相同的pair会被计算多次
            tp, fn, fp = _compute_tp_fn_fp(set(list(oe_pred_counter.keys())),
                                           set(list(oe_target_counter.keys())))
            self.oe_fn += fn
            self.oe_fp += fp
            self.oe_tp += tp

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1,self.oe_tp, self.oe_fn, self.oe_fp)

        res['oe_f'] = round(f*100, 2)
        res['oe_rec'] = round(rec*100, 2)
        res['oe_pre'] = round(pre*100, 2)

        res['em'] = round(self.em/self.total, 4)
        res['invalid'] = round(self.invalid/self.total, 4)
        if reset:
            self.oe_fp = 0
            self.oe_tp = 0
            self.oe_fn = 0

        return res


class AESCSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, conflict_id, opinion_first=True):
        super(AESCSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.word_start_index = num_labels+2

        self.aesc_fp = 0
        self.aesc_tp = 0
        self.aesc_fn = 0
        self.ae_fp = 0
        self.ae_tp = 0
        self.ae_fn = 0
        self.sc_fp = Counter()
        self.sc_tp = Counter()
        self.sc_fn = Counter()

        self.em = 0
        self.total = 0
        self.invalid = 0
        self.conflict_id = conflict_id
        # assert opinion_first is False, "Current metric only supports aspect first"

    def evaluate(self, aesc_target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len-2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(aesc_target_span, pred.tolist())):
            em = 0
            assert ps[0]==tgt_tokens[i, 0]
            ps = ps[2:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps):
                    if j<self.word_start_index:
                        cur_pair.append(j)
                        if len(cur_pair)!=3 or cur_pair[0]>cur_pair[1]:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())
            self.invalid += invalid

            aesc_target_counter = Counter()
            aesc_pred_counter = Counter()
            ae_target_counter = Counter()
            ae_pred_counter = Counter()
            conflicts = set()

            for t in ts:
                ae_target_counter[(t[0], t[1])] = 1
                if t[2]!=self.conflict_id:
                    aesc_target_counter[(t[0], t[1])] = t[2]
                else:
                    conflicts.add((t[0], t[1]))

            for p in pairs:
                ae_pred_counter[(p[0], p[1])] = 1
                if (p[0], p[1]) not in conflicts and p[-1] not in (0, 1, self.conflict_id):
                    aesc_pred_counter[(p[0], p[1])] = p[-1]

            # 这里相同的pair会被计算多次
            tp, fn, fp = _compute_tp_fn_fp([(key[0], key[1], value) for key,value in aesc_pred_counter.items()],
                                           [(key[0], key[1], value) for key,value in aesc_target_counter.items()])
            self.aesc_fn += fn
            self.aesc_fp += fp
            self.aesc_tp += tp

            tp, fn, fp = _compute_tp_fn_fp(list(aesc_pred_counter.keys()),
                                           list(aesc_target_counter.keys()))
            self.ae_fn += fn
            self.ae_fp += fp
            self.ae_tp += tp

            # sorry, this is a very wrongdoing, but to make it comparable with previous work, we have to stick to the
            #   error
            for key in aesc_pred_counter:
                if key not in aesc_target_counter:
                    continue
                if aesc_target_counter[key]==aesc_pred_counter[key]:
                    self.sc_tp[aesc_pred_counter[key]] += 1
                    aesc_target_counter.pop(key)
                else:
                    self.sc_fp[aesc_pred_counter[key]] += 1
                    self.sc_fn[aesc_target_counter[key]] += 1

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1,self.aesc_tp, self.aesc_fn, self.aesc_fp)
        res['aesc_f'] = round(f*100, 2)
        res['aesc_rec'] = round(rec*100, 2)
        res['aesc_pre'] = round(pre*100, 2)

        f, pre, rec = _compute_f_pre_rec(1, self.ae_tp, self.ae_fn, self.ae_fp)
        res['ae_f'] = round(f*100, 2)
        res['ae_rec'] = round(rec*100, 2)
        res['ae_pre'] = round(pre*100, 2)

        tags = set(self.sc_tp.keys())
        tags.update(set(self.sc_fp.keys()))
        tags.update(set(self.sc_fn.keys()))
        f_sum = 0
        pre_sum = 0
        rec_sum = 0
        for tag in tags:
            assert tag not in (0, 1, self.conflict_id), (tag, self.conflict_id)
            tp = self.sc_tp[tag]
            fn = self.sc_fn[tag]
            fp = self.sc_fp[tag]
            f, pre, rec = _compute_f_pre_rec(1, tp, fn, fp)
            f_sum += f
            pre_sum += pre
            rec_sum += rec

        rec_sum /= (len(tags)+1e-12)
        pre_sum /= (len(tags)+1e-12)
        res['sc_f'] = round(2*pre_sum*rec_sum/(pre_sum+rec_sum+1e-12)*100, 2)
        res['sc_rec'] = round(rec_sum*100, 2)
        res['sc_pre'] = round(pre_sum*100, 2)

        res['em'] = round(self.em/self.total, 4)
        res['invalid'] = round(self.invalid/self.total, 4)
        if reset:
            self.aesc_fp = 0
            self.aesc_tp = 0
            self.aesc_fn = 0
            self.ae_fp = 0
            self.ae_tp = 0
            self.ae_fn = 0
            self.sc_fp = Counter()
            self.sc_tp = Counter()
            self.sc_fn = Counter()

        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, (list, set)):
        ts = {key:1 for key in list(ts)}
    if isinstance(ps, (list, set)):
        ps = {key:1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp

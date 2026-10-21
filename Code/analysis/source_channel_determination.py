import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

SR_DIR = r"<PROJECT_DIR>\<TIMESTAMPS_DIR>"
LABEL_I = r"<DATA_DIR>\<DATA_DIR>\labels"
SR_COOC = 0.035
CLUSTER_WIN = 0.05
SR_TH = 1.40

def per_channel_source(spike_pos, rs_list, re_list):
    
    n_ch = len(rs_list)
    events = []
    sr_count = np.zeros(n_ch)
    for ci in range(n_ch):
        sp = np.sort(np.asarray(spike_pos[ci], float))
        rs = np.asarray(rs_list[ci], float); re_ = np.asarray(re_list[ci], float)
        if len(rs) == 0 or len(sp) == 0:
            continue
        rmid = (rs + re_) / 2.0
        is_sr = np.array([np.any(np.abs(sp - m) <= SR_COOC) for m in rmid])
        sr_count[ci] = int(is_sr.sum())
        for t in rs[is_sr]:
            events.append((t, ci))
    if len(events) < 2:
        return None
    events.sort(key=lambda x: x[0])
    clusters = []; cur = [events[0]]
    for e in events[1:]:
        if e[0] - cur[-1][0] <= CLUSTER_WIN:
            cur.append(e)
        else:
            clusters.append(cur); cur = [e]
    clusters.append(cur)
    clusters = [c for c in clusters if len(c) >= 2]
    if not clusters:
        return None
    src_count = np.zeros(n_ch); part_count = np.zeros(n_ch)
    for c in clusters:
        c.sort(key=lambda x: x[0])
        src_count[c[0][1]] += 1
        for e in c:
            part_count[e[1]] += 1
    source_freq = np.divide(src_count, part_count, out=np.zeros_like(src_count), where=part_count > 0)
    return source_freq, sr_count

def core_source_enrich(spike_pos, rs_list, re_list, core_mask):
    
    n_ch = len(core_mask)
    events = []
    for ci in range(n_ch):
        sp = np.sort(np.asarray(spike_pos[ci], float))
        rs = np.asarray(rs_list[ci], float); re_ = np.asarray(re_list[ci], float)
        if len(rs) == 0 or len(sp) == 0:
            continue
        rmid = (rs + re_) / 2.0
        is_sr = np.array([np.any(np.abs(sp - m) <= SR_COOC) for m in rmid])
        for t in rs[is_sr]:
            events.append((t, ci))
    if len(events) < 2:
        return np.nan
    events.sort(key=lambda x: x[0])
    clusters = []; cur = [events[0]]
    for e in events[1:]:
        if e[0] - cur[-1][0] <= CLUSTER_WIN:
            cur.append(e)
        else:
            clusters.append(cur); cur = [e]
    clusters.append(cur)
    clusters = [c for c in clusters if len(c) >= 2]
    if not clusters:
        return np.nan
    n_src_core = sum(1 for c in clusters if core_mask[c[0][1]])
    core_src_rate = n_src_core / len(clusters)
    core_event_rate = sum(1 for e in events if core_mask[e[1]]) / len(events)
    return core_src_rate - core_event_rate

def main():
    data = {}
    for pt in sorted(os.listdir(SR_DIR)):
        sf = os.path.join(SR_DIR, f'{pt}_sr.npz')
        if not os.path.isfile(sf):
            continue
        z = np.load(sf, allow_pickle=True)
        ch = list(z['ch_names']); dur = float(z['dur_s']); n = len(ch)
        ez = None
        if ez is None:
            _e = z['ez_clin']
            ez = np.asarray(_e, int) if (_e is not None and np.asarray(_e).ndim > 0) else np.zeros(n, int)
        spike_pos = [np.asarray(z['spike_pos'][i], float) for i in range(n)]
        rs = [np.asarray(z['ripple_start'][i], float) for i in range(n)]
        re_ = [np.asarray(z['ripple_end'][i], float) for i in range(n)]
        res = per_channel_source(spike_pos, rs, re_)
        if res is None:
            continue
        source_freq, sr_count = res
        sr_rate = sr_count / dur * 60.0
        core_mask = sr_rate >= SR_TH
        enrich = core_source_enrich(spike_pos, rs, re_, core_mask)

        valid = sr_count > 0
        rho = np.nan; auc = np.nan
        if valid.sum() >= 5:
            sf = source_freq[valid]; sr = sr_rate[valid]; e = ez[valid]
            rho, _ = spearmanr(sr, sf)
            if e.sum() > 0 and (e == 0).sum() > 0:
                auc = roc_auc_score(e, sf)
        data[pt] = {'enrich': enrich, 'rho': rho, 'auc': auc, 'grade': 0}
    pts = list(data.keys())
    print(f'cohort {len(pts)} cases (Engel I={sum(1 for p in pts if data[p]["grade"]==1)}, II-IV={sum(1 for p in pts if data[p]["grade"]==0)})\n')

    def report(label, vals, thresh_test):
        v = np.array(vals); v = v[~np.isnan(v)]
        if len(v) < 3:
            print(f'  {label}: n={len(v)} too few')
            return
        med = np.median(v); pos = (v > thresh_test).mean() * 100
        p = wilcoxon(v - thresh_test).pvalue
        print(f'  {label}: median={med:+.3f}  {">" if thresh_test==0 else ">"}{thresh_test}ratio of cases={pos:.0f}%  Wilcoxon p={p:.4f}  n={len(v)}')

    for grp, mask in [('Engel I', lambda p: data[p]['grade'] == 1),
                      ('II-IV', lambda p: data[p]['grade'] == 0)]:
        print(f'=== {grp} ===')
        report('onset enrichment (core as source−baseline)', [data[p]['enrich'] for p in pts if mask(p)], 0)
        report('Spearman(SR rate→sourcefrequency)', [data[p]['rho'] for p in pts if mask(p)], 0)
        report('AUC(source→EZ/resection)', [data[p]['auc'] for p in pts if mask(p)], 0.5)
        print()

    pd.DataFrame([{'patient': p, **data[p]} for p in pts]).to_csv(
        r'<PROJECT_DIR>\sr_part4_70th.csv', index=False, encoding='utf-8-sig')
    print('saved → sr_part4_70th.csv')

if __name__ == '__main__':
    main()

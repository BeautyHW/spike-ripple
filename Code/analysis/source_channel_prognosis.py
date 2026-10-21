import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

SR_DIR = r"<PROJECT_DIR>\<TIMESTAMPS_DIR>"
LABEL_I = r"<DATA_DIR>\<DATA_DIR>\labels"
SR_COOC = 0.035
CLUSTER_WIN = 0.05

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

def main():
    data = {}
    for pt in sorted(os.listdir(SR_DIR)):
        sf = os.path.join(SR_DIR, f'{pt}_sr.npz')
        if not os.path.isfile(sf):
            continue
        z = np.load(sf, allow_pickle=True)
        ch = list(z['ch_names']); n = len(ch)
        ez = None
        if ez is None:
            _e = z['ez_clin']
            ez = np.asarray(_e, int) if (_e is not None and np.asarray(_e).ndim > 0) else np.zeros(n, int)
        res = per_channel_source([np.asarray(z['spike_pos'][i], float) for i in range(n)],
                                 [np.asarray(z['ripple_start'][i], float) for i in range(n)],
                                 [np.asarray(z['ripple_end'][i], float) for i in range(n)])
        if res is None:
            continue
        source_freq, sr_count = res
        data[pt] = {'source_freq': source_freq, 'sr_count': sr_count, 'ez': ez,
                    'grade': 0}
    pts = list(data.keys())
    grades = np.array([data[p]['grade'] for p in pts])
    print(f'cohort {len(pts)} cases (Engel I={int(grades.sum())}, II-IV={int((1-grades).sum())})\n')

    comps = {}
    for p in pts:
        sf = data[p]['source_freq']; ez = data[p]['ez']
        src = sf > 0
        if src.sum() == 0:
            comps[p] = np.nan
        else:
            comps[p] = (src & (ez == 1)).sum() / src.sum()

    def eval_auc(c, g):
        c = np.asarray(c); g = np.asarray(g); ok = ~np.isnan(c); c = c[ok]; g = g[ok]
        if len(c) < 5 or g.sum() == 0 or g.sum() == len(g):
            return None
        auc = roc_auc_score(g, c)
        core = (c > 0.5).astype(int)
        tp = ((core == 1) & (g == 1)).sum(); fp = ((core == 1) & (g == 0)).sum()
        fn = ((core == 0) & (g == 1)).sum(); tn = ((core == 0) & (g == 0)).sum()
        ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
        or_, p = fisher_exact([[tp, fp], [fn, tn]])
        return dict(AUC=auc, PPV=ppv, NPV=npv, OR=or_, p=p, n=len(c), sens=tp/max(tp+fn,1), spec=tn/max(tn+fp,1))

    c_all = np.array([comps[p] for p in pts])
    ins = eval_auc(c_all, grades)
    print('=== source resectionfraction -> prognosis (in-sample)===')
    if ins:
        print(f'AUC={ins["AUC"]:.3f}  sens={ins["sens"]:.3f}  spec={ins["spec"]:.3f}  '
              f'PPV={ins["PPV"]:.3f}  NPV={ins["NPV"]:.3f}  OR={ins["OR"]:.1f}  p={ins["p"]:.5f}  n={ins["n"]}')

    pred = []
    for ti, p_test in enumerate(pts):
        ok = [j for j in range(len(pts)) if j != ti and not np.isnan(comps[pts[j]])]
        if len(ok) < 5:
            continue

        pred.append(comps[p_test])

    rng = np.random.default_rng(20260817)
    ok_all = ~np.isnan(c_all); c = c_all[ok_all]; g = grades[ok_all]
    aucs = []
    for _ in range(1000):
        idx = rng.integers(0, len(c), len(c)); gi = g[idx]; ci = c[idx]
        if gi.sum() > 0 and gi.sum() < len(gi) and len(np.unique(ci)) > 1:
            aucs.append(roc_auc_score(gi, ci))
    print(f'\nAUC bootstrap SD = {np.std(aucs):.3f} (patientslevel 1000 times)')

    print(f'\n=== source resectionfraction by outcome (median)===')
    for label, mask in [('Engel I', grades == 1), ('II-IV', grades == 0)]:
        v = c_all[mask]; v = v[~np.isnan(v)]
        print(f'  {label}: median={np.median(v):.3f}  mean={v.mean():.3f}  n={len(v)}')

    rows = [{'patient': p, 'grade': data[p]['grade'], 'source_resect_frac': comps[p]} for p in pts]
    pd.DataFrame(rows).to_csv(r'<PROJECT_DIR>\sr_source_prognosis.csv', index=False, encoding='utf-8-sig')
    print('\nsaved → sr_source_prognosis.csv')

if __name__ == '__main__':
    main()

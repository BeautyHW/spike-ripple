import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

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

def main():
    rows = []
    for pt in sorted(os.listdir(SR_DIR)):
        sf = os.path.join(SR_DIR, f'{pt}_sr.npz')
        if not os.path.isfile(sf):
            continue
        z = np.load(sf, allow_pickle=True)
        ch = list(z['ch_names']); dur = float(z['dur_s']); n = len(ch)
        res = per_channel_source([np.asarray(z['spike_pos'][i], float) for i in range(n)],
                                 [np.asarray(z['ripple_start'][i], float) for i in range(n)],
                                 [np.asarray(z['ripple_end'][i], float) for i in range(n)])
        if res is None:
            continue
        source_freq, sr_count = res
        sr_rate = sr_count / dur * 60.0
        source = source_freq > 0
        dense = sr_rate >= SR_TH
        n_src = int(source.sum()); n_dense = int(dense.sum())
        n_over = int((source & dense).sum())
        union = n_src + n_dense - n_over
        jaccard = n_over / union if union > 0 else np.nan
        src_in_dense = n_over / n_src if n_src > 0 else np.nan
        dense_in_src = n_over / n_dense if n_dense > 0 else np.nan
        rows.append({'patient': pt, 'grade': 0,
                     'n_source': n_src, 'n_dense': n_dense, 'n_overlap': n_over,
                     'jaccard': jaccard, 'src_in_dense': src_in_dense, 'dense_in_src': dense_in_src})
    df = pd.DataFrame(rows)
    df.to_csv(r'<PROJECT_DIR>\sr_source_dense_overlap.csv', index=False, encoding='utf-8-sig')

    print(f'=== {len(df)} cases overall ===')
    print(f'Jaccard overlap: median={df.jaccard.median():.3f}  mean={df.jaccard.mean():.3f}')
    print(f'source channels in dense region ratio of cases (src_in_dense): median={df.src_in_dense.median():.3f}  mean={df.src_in_dense.mean():.3f}')
    print(f'dense channels that are source ratio of cases (dense_in_src): median={df.dense_in_src.median():.3f}  mean={df.dense_in_src.mean():.3f}')

    tot_src = df.n_source.sum(); tot_dense = df.n_dense.sum(); tot_over = df.n_overlap.sum()
    print(f'pooled Jaccard = {tot_over/(tot_src+tot_dense-tot_over):.3f}  (source{int(tot_src)} / dense{int(tot_dense)} / overlap{int(tot_over)})')

    print('\n=== by outcome ===')
    for label, mask in [('Engel I', df.grade == 1), ('II-IV', df.grade == 0)]:
        sub = df[mask]
        print(f'{label} (n={len(sub)}): Jaccard median={sub.jaccard.median():.3f}  '
              f'src_in_dense median={sub.src_in_dense.median():.3f}  dense_in_src median={sub.dense_in_src.median():.3f}')

    print('\nper patient (first 20 rows preview):')
    print(df[['patient','grade','n_source','n_dense','n_overlap','jaccard']].head(20).to_string(index=False))
    print('\nsaved → sr_source_dense_overlap.csv')

if __name__ == '__main__':
    main()

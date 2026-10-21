import os, argparse, glob, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import mne
from scipy.signal import butter, sosfilt
from scipy.stats import wilcoxon
from scipy.linalg import toeplitz

FS = 500
ORDER = 50
SR_TH = 1.40
SR_WIN = 0.035
RIDGE = 1e-2


def sr_rate(spike_pos, rs, re_, dur, win=SR_WIN):
    sp = np.sort(np.asarray(spike_pos, float)); rs = np.asarray(rs, float); re_ = np.asarray(re_, float)
    if len(rs) == 0 or len(sp) == 0:
        return 0.0
    rpk = (rs + re_) / 2; idx = np.searchsorted(sp, rpk)
    best = np.full(len(rpk), np.inf); bs = np.zeros(len(rpk))
    for off in (0, -1):
        j = idx + off; v = (j >= 0) & (j < len(sp)); jc = np.clip(j, 0, len(sp) - 1)
        d = np.abs(rpk - sp[jc]); d = np.where(v, d, np.inf); m = d < best; best[m] = d[m]; bs[m] = sp[jc][m]
    return len(bs[best <= win]) / dur * 60.0

def sr_event_times(spike_pos, rs, re_, win=SR_WIN):
    
    sp = np.sort(np.asarray(spike_pos, float)); rs = np.asarray(rs, float); re_ = np.asarray(re_, float)
    if len(rs) == 0 or len(sp) == 0:
        return np.array([])
    rpk = (rs + re_) / 2
    is_sr = np.array([np.any(np.abs(sp - m) <= win) for m in rpk])
    return rpk[is_sr]

def _yw2(A_r, order, alpha=RIDGE):
    T = np.zeros((2 * order, 2 * order))
    for m in range(order):
        Sm = np.diag(np.ones(order - m), -m)
        T += np.kron(Sm, A_r[m])
        if m > 0:
            T += np.kron(Sm.T, A_r[m].T)
    B = A_r[1:].reshape(order * 2, 2)
    sol = np.linalg.solve(T + alpha * np.eye(order * 2), B)
    A = sol.reshape(order, 2, 2).transpose(1, 0, 2)
    Sig = A_r[0].copy()
    for k in range(order):
        Sig -= A[:, k, :] @ A_r[k + 1].T
    return (Sig + Sig.T) / 2

def cross_region_gc(data, order, core_idx, rest_idx):
    n_ch, n = data.shape
    x = data - data.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-12)
    R = np.zeros((order + 1, n_ch, n_ch))
    for tau in range(order + 1):
        R[tau] = x[:, tau:] @ x[:, :n - tau].T / (n - tau)
    sigma_r = np.zeros(n_ch)
    for c in range(n_ch):
        r = R[:, c, c]
        T = toeplitz(r[:order])
        sol = np.linalg.solve(T + RIDGE * np.eye(order), r[1:])
        sigma_r[c] = max(r[0] - np.dot(sol, r[1:]), 1e-9)
    out_list, in_list = [], []
    for i in core_idx:
        for j in rest_idx:
            Sig = _yw2(R[:, [i, j]][:, :, [i, j]], order)
            gc_core_to_rest = np.log(sigma_r[j] / max(Sig[1, 1], 1e-9))
            gc_rest_to_core = np.log(sigma_r[i] / max(Sig[0, 0], 1e-9))
            out_list.append(gc_core_to_rest)
            in_list.append(gc_rest_to_core)
    return np.mean(out_list), np.mean(in_list)

def build_sr_clean(data, sr_times_union, fs, n_samp, pre_ms, post_ms):
    pre, post = int(pre_ms / 1000 * fs), int(post_ms / 1000 * fs)
    st = np.sort(sr_times_union)
    s_win = [(max(0, int(round(t * fs)) - pre), int(round(t * fs)) + post) for t in st
             if int(round(t * fs)) + post <= n_samp and int(round(t * fs)) - pre >= 0]
    c_win = []; prev = 0
    for a, b in s_win:
        if a - prev >= int(1.0 * fs):
            c_win.append((prev, a))
        prev = max(prev, b)
    if n_samp - prev >= int(1.0 * fs):
        c_win.append((prev, n_samp))
    taper = np.ones(pre + post); e = 10
    taper[:e] = np.sin(np.pi * np.arange(e) / (2 * e)); taper[-e:] = np.sin(np.pi * (np.arange(e) + e) / (2 * e))
    sr_c = np.concatenate([data[:, a:b] * taper[None, :] for a, b in s_win], axis=1) if s_win else np.zeros((data.shape[0], 1))
    clean_c = np.concatenate([data[:, a:b] for a, b in c_win], axis=1) if c_win else np.zeros((data.shape[0], 1))
    MAX = 50000
    if sr_c.shape[1] > MAX:
        sr_c = sr_c[:, :MAX]
    if clean_c.shape[1] > MAX:
        clean_c = clean_c[:, :MAX]
    return sr_c, clean_c

def main():
    ap = argparse.ArgumentParser(description='II-IV group SR window Granger (core→rest net flow out)')
    ap.add_argument('--npz_dir', required=True, help='SR detection npz directory (with spike_pos/ripple_start/ripple_end)')
    ap.add_argument('--fif_dir', required=True, help='raw interictal FIF directory')
    ap.add_argument('--out', default='sr_gc_iiv.csv', help='output CSV path')
    a = ap.parse_args()
    pts = sorted(os.listdir(a.npz_dir))
    print(f'target {len(pts)} cases II-IV\n')

    rows = []
    t0 = time.time()
    for pt in pts:
        npz = os.path.join(a.npz_dir, f'{pt}_sr.npz')
        if not os.path.isfile(npz):
            print(f'{pt}: no npz', flush=True); continue
        z = np.load(npz, allow_pickle=True)
        ch = list(z['ch_names']); dur = float(z['dur_s'])
        fifs = (glob.glob(os.path.join(a.fif_dir, f'{pt}_*.fif')) or
                glob.glob(os.path.join(a.fif_dir, f'{pt[:-1]}_*.fif')) or
                glob.glob(os.path.join(a.fif_dir, f'{pt}', '*.fif')) or
                glob.glob(os.path.join(a.fif_dir, '**', f'{pt}_*.fif'), recursive=True))
        if not fifs:
            print(f'{pt}: no FIF', flush=True); continue
        raw = mne.io.read_raw_fif(fifs[0], preload=True, verbose='error')
        raw.resample(FS, npad='auto', verbose='error')
        data = raw.get_data().astype(np.float64)
        sos = butter(4, [1, 80], btype='bandpass', fs=FS, output='sos')
        data = sosfilt(sos, data, axis=1)
        n_samp = data.shape[1]
        sr = np.zeros(len(ch)); sr_union = []
        for ci in range(len(ch)):
            sp = np.asarray(z['spike_pos'][ci], float)
            rs = np.asarray(z['ripple_start'][ci], float); re_ = np.asarray(z['ripple_end'][ci], float)
            st = sr_event_times(sp, rs, re_)
            sr_union.extend(st.tolist())
            sr[ci] = sr_rate(sp, rs, re_, dur)
        core_idx = np.where(sr >= SR_TH)[0]; rest_idx = np.where(sr < SR_TH)[0]
        if len(core_idx) == 0 or len(rest_idx) == 0:
            print(f'{pt}: no core or rest', flush=True); continue
        row = {'patient': pt, 'n_core': len(core_idx)}
        for pre_ms, post_ms in [(50, 150), (100, 300), (200, 500)]:
            sr_c, clean_c = build_sr_clean(data, np.asarray(sr_union), FS, n_samp, pre_ms, post_ms)
            out_s, in_s = cross_region_gc(sr_c, ORDER, core_idx, rest_idx)
            out_c, in_c = cross_region_gc(clean_c, ORDER, core_idx, rest_idx)
            row[f'sr_net_{pre_ms}_{post_ms}'] = out_s - in_s
            row[f'clean_net_{pre_ms}_{post_ms}'] = out_c - in_c
        rows.append(row)
        print(f'{pt}: SRnet flow(100,300)={row["sr_net_100_300"]:+.3f} cleannet flow={row["clean_net_100_300"]:+.3f} (core={len(core_idx)}) {time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, encoding='utf-8-sig')
    print(f'\n=== {len(df)} cases saved → {a.out} ===')
    for pre_ms, post_ms in [(50, 150), (100, 300), (200, 500)]:
        sr_col = f'sr_net_{pre_ms}_{post_ms}'
        clean_col = f'clean_net_{pre_ms}_{post_ms}'
        d = df[sr_col] - df[clean_col]
        print(f'window ({pre_ms},{post_ms}): SRnet flowmedian={df[sr_col].median():+.4f}  >0ratio of cases={(df[sr_col]>0).mean()*100:.0f}%  '
              f'cleannet flowmedian={df[clean_col].median():+.4f}  SR-clean Wilcoxon p={wilcoxon(d).pvalue:.4f}')
    print(f'\ntotal time {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()

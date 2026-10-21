import os, sys, glob, warnings, time, pickle
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import mne
sys.path.insert(0, r'<PROJECT_DIR>')
from janca_detector import spike_detector, prepare_from_fif

SR_DIR = r'<PROJECT_DIR>\<TIMESTAMPS_DIR>'
FIF_DIRS = [r'<DATA_DIR>\<PREPROCESSED_DIR>',
            r'<DATA_DIR>\<PREPROCESSED_DIR>']
LABEL_DIR = r'<DATA_DIR>\<DATA_DIR>\labels'
OUT = r'<PROJECT_DIR>'
COOC_WIN = 0.035
JANCA_K1 = 3.65


def find_fif(pt):
    for dd in FIF_DIRS:
        ptd = os.path.join(dd, pt)
        if os.path.isdir(ptd):
            fifs = sorted(glob.glob(os.path.join(ptd, '*_300s_raw.fif')))
            if fifs:
                return fifs[0]
    return None

def load_ez(pt, ch_names):
    lf = os.path.join(LABEL_DIR, f'{pt}_labels.csv')
    if not os.path.isfile(lf):
        return np.zeros(len(ch_names), int)
    dl = pd.read_csv(lf, encoding='utf-8-sig')
    col = 'channel_name' if 'channel_name' in dl.columns else 'channel'
    em = dict(zip(dl[col], dl['EZ_NEZ']))
    return np.array([int(em.get(c, 0)) for c in ch_names])

def count_spikes_per_ripple(spike_pos, rs, re_, win=COOC_WIN):
    sp = np.sort(np.asarray(spike_pos, float))
    rs = np.asarray(rs, float); re_ = np.asarray(re_, float)
    if len(sp) == 0 or len(rs) == 0:
        return np.zeros(len(rs), int)
    rmid = (rs + re_) / 2.0
    cnts = np.zeros(len(rmid), int)
    for i, m in enumerate(rmid):
        lo = np.searchsorted(sp, m - win)
        hi = np.searchsorted(sp, m + win, side='right')
        cnts[i] = hi - lo
    return cnts

def process(pt):
    fif = find_fif(pt)
    npz = os.path.join(SR_DIR, f'{pt}_sr.npz')
    if fif is None or not os.path.isfile(npz):
        return None
    z = np.load(npz, allow_pickle=True)
    ch = list(z['ch_names']); dur = float(z['dur_s'])
    n = len(ch)
    ez = load_ez(pt, ch)

    data_j, ch_j, fs_j = prepare_from_fif(fif, fs_target=250)
    spikes, _ = spike_detector(data_j, fs_j, k1=JANCA_K1, polyspike_union_time=0.0)
    spike_pos = [np.asarray(spikes['pos'][spikes['chan'] == c], float) for c in range(n)]

    single_rate = np.zeros(n); poly_rate = np.zeros(n)
    for ci in range(n):
        rs = np.asarray(z['ripple_start'][ci], float)
        re_ = np.asarray(z['ripple_end'][ci], float)
        cnts = count_spikes_per_ripple(spike_pos[ci], rs, re_)
        single_rate[ci] = (cnts == 1).sum() * 60 / dur
        poly_rate[ci] = (cnts >= 2).sum() * 60 / dur

    return {'ch': ch, 'ez': ez, 'single_rate': single_rate, 'poly_rate': poly_rate, 'dur': dur}

def main():
    t0 = time.time()
    cache = {}
    for pt in sorted(os.listdir(NPZ_DIR)):
        try:
            r = process(pt)
        except Exception as e:
            print(f'[ERR] {pt}: {str(e)[:100]}')
            continue
        if r is None:
            print(f'[SKIP] {pt}')
            continue
        cache[pt] = r
        print(f'{pt}: single SR={int(r["single_rate"].sum()*r["dur"]/60)} poly SR={int(r["poly_rate"].sum()*r["dur"]/60)} '
              f'(EZ median single={np.median(r["single_rate"][r["ez"]==1]):.2f} poly={np.median(r["poly_rate"][r["ez"]==1]):.2f}/min) ({time.time()-t0:.0f}s)')

    with open(os.path.join(OUT, 'part7_multiplicity_cache.pkl'), 'wb') as f:
        pickle.dump(cache, f)
    print(f'\ncached {len(cache)} cases → part7_multiplicity_cache.pkl')

    ez_single, nez_single, ez_poly, nez_poly = [], [], [], []
    for pt, r in cache.items():
        ez = r['ez'] == 1
        ez_single.extend(r['single_rate'][ez]); nez_single.extend(r['single_rate'][~ez])
        ez_poly.extend(r['poly_rate'][ez]); nez_poly.extend(r['poly_rate'][~ez])
    print('\n=== channel-level EZ vs NEZ (median rate /min)===')
    print(f'  single-spike SR: EZmedian={np.median(ez_single):.3f}  NEZmedian={np.median(nez_single):.3f}  ratio ={np.median(ez_single)/max(np.median(nez_single),1e-9):.2f}')
    print(f'  poly-spike SR: EZmedian={np.median(ez_poly):.3f}  NEZmedian={np.median(nez_poly):.3f}  ratio ={np.median(ez_poly)/max(np.median(nez_poly),1e-9):.2f}')

if __name__ == '__main__':
    main()

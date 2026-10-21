r
import os, glob, time
import numpy as np
import pandas as pd
import mne
import warnings
from scipy.signal import butter, filtfilt, iirnotch, convolve2d
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

ICTAL_ROOT = r"<DATA_DIR>\part12_ictal"
MAP_CSV = os.path.join(ICTAL_ROOT, "part12_channel_map.csv")
SOZ_CSV = os.path.join(ICTAL_ROOT, "part12_soz_annotation.csv")
OUT_DIR = os.path.join(ICTAL_ROOT, "out")

BASE_WIN = (30.0, 10.0)
TARGET_WIN = (10.0, 20.0)
BAND = (60.0, 140.0)
THR_SD = 20
SUSTAIN = 0.25
ER_BANDS = [(3.5, 7.4), (7.4, 12.4), (12.4, 24.0), (24.0, 97.0)]
ER_H = 5.0
ER_EPS = 1.0
MAD_K = 10.0

REF_WIN = 10.0
MIN_REF = 8.0
CORROB_WIN = 2.0
MIN_CORROB = 2
REAN_TOL = 5.0
MIN_BASE = 10.0
MIN_TARGET = 15.0
SYNC_FRAC = 0.8
SYNC_SPAN = 1.0
SUS_ELEV = 2.0

def compute_hfer(target_data, base_data, fs):
    target_sq = target_data ** 2
    base_sq = base_data ** 2
    window = int(fs / 2.0)
    target_energy = convolve2d(target_sq, np.ones((1, window)), 'same')
    base_energy = convolve2d(base_sq, np.ones((1, window)), 'same')
    base_energy_ref = np.sum(base_energy, axis=1) / base_energy.shape[1]
    norm_target_energy = target_energy / base_energy_ref[:, np.newaxis]
    norm_base_energy = base_energy / base_energy_ref[:, np.newaxis]
    return norm_target_energy, norm_base_energy

def bq_thresholds(norm_b, thr_sd=THR_SD):
    return np.max(norm_b, axis=1) + thr_sd * np.std(norm_b, axis=1, ddof=1)

def sustained_onset(norm_t, norm_b, fs, mode='mad', sustain=SUSTAIN):
    
    n_ch, n_t = norm_t.shape
    onset = np.full(n_ch, n_t, dtype=int)
    w = int(sustain * fs)
    if w < 1 or n_t <= w or norm_b.shape[1] <= w:
        return onset, np.full(n_ch, np.nan)

    def rolling_means(x):
        c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
        return (c[:, w:] - c[:, :-w]) / w

    mb = rolling_means(norm_b)
    if mode == 'p99':
        thr = np.percentile(mb, 99, axis=1)
    else:
        thr = mad_threshold(mb)
    mt = rolling_means(norm_t)
    above = mt > thr[:, np.newaxis]
    any_ = above.any(axis=1)
    onset[any_] = np.argmax(above, axis=1)[any_]
    return onset, thr

def mad_threshold(x, k=MAD_K):
    med = np.median(x, axis=1)
    mad = np.median(np.abs(x - med[:, np.newaxis]), axis=1)
    return med + k * 1.4826 * mad

def bq_ei(norm_t, onset_idx, fs, self_anchor=False):
    n_t = norm_t.shape[1]
    seizure_location = int(onset_idx.min())
    w = int(0.25 * fs)
    if self_anchor:
        hfer = np.full(norm_t.shape[0], np.nan)
        for i in range(norm_t.shape[0]):
            if onset_idx[i] < n_t:
                a = onset_idx[i]
                hfer[i] = np.nanmean(norm_t[i, a:min(a + w, n_t)])
    else:
        a = seizure_location
        seg = norm_t[:, a:min(a + w, n_t)]
        hfer = np.nanmean(seg, axis=1)
    order = np.argsort(onset_idx, kind='stable')
    rank = np.empty(norm_t.shape[0])
    rank[order] = np.arange(1, norm_t.shape[0] + 1)
    ei = np.sqrt(hfer / rank)
    ei[~np.isfinite(ei)] = 0.0
    if np.max(ei) > 0:
        ei = ei / np.max(ei)
    return ei, hfer

def er_series(data, fs):
    w = int(fs * 0.5)
    energies = []
    for lo, hi in ER_BANDS:
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
        xf = filtfilt(b, a, data, axis=-1)
        energies.append(convolve2d(xf ** 2, np.ones((1, w)), 'same'))
    theta, alpha, beta, gamma = energies
    return (beta + gamma) / (theta + alpha + 1e-20)

def er_ei(er_t, onset_idx, fs):
    n_ch, n_t = er_t.shape
    t_min = onset_idx.min()
    ei = np.zeros(n_ch)
    for i in range(n_ch):
        if onset_idx[i] < n_t:
            a, b = onset_idx[i], min(onset_idx[i] + int(ER_H * fs), n_t)
            energy = np.nanmean(er_t[i, a:b]) if b > a else np.nan
            delay = (onset_idx[i] - t_min) / fs + ER_EPS
            if np.isfinite(energy):
                ei[i] = energy / delay
    if np.max(ei) > 0:
        ei = ei / np.max(ei)
    return ei

def brainquake_filter(data, fs, band=BAND):
    data = data - np.mean(data, axis=0)
    for nf in (50.0, 100.0, 150.0):
        tb, ta = iirnotch(nf / (fs / 2), 30)
        data = filtfilt(tb, ta, data, axis=-1)
    nyq = fs / 2
    b, a = butter(5, np.array([band[0] / nyq, band[1] / nyq]), btype='bandpass')
    data = filtfilt(b, a, data)
    return data

def norm_ch(name):
    return str(name).replace(" ", "").translate({ord(c): ord("'") for c in "`´’"})

def flags_for(pt, ch_names, ch_map_norm, col):
    g = ch_map_norm.get(pt)
    if g is None:
        return np.full(len(ch_names), np.nan)
    idx, collide = {}, set()
    for ch in g.index:
        v = g.loc[ch, col]
        k = norm_ch(ch)
        if k in idx and idx[k] != v:
            collide.add(k)
        idx[k] = v
    for k in collide:
        idx[k] = np.nan
    return np.array([idx.get(norm_ch(c), np.nan) for c in ch_names], float)

def zs(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)

def stage1_onset(scan, ref, fs):
    
    n_ch, n_t = scan.shape
    onset = np.full(n_ch, n_t, dtype=int)
    w = int(SUSTAIN * fs)
    if w < 1 or n_t <= w or ref.shape[1] <= w:
        return onset

    def rolling_means(x):
        c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
        return (c[:, w:] - c[:, :-w]) / w

    thr = mad_threshold(rolling_means(ref))
    mt = rolling_means(scan)
    above = (mt > thr[:, np.newaxis]).astype(np.int32)
    run = int(SUS_ELEV * fs)
    if above.shape[1] < run:
        return onset
    csum = np.concatenate([np.zeros((n_ch, 1), np.int32), np.cumsum(above, axis=1)], axis=1)
    sums = csum[:, run:] - csum[:, :-run]
    ok = sums == run
    any_ = ok.any(axis=1)
    onset[any_] = np.argmax(ok, axis=1)[any_]
    return onset

def corroborated_earliest(onset_idx, ref1, fs, n_scan):
    
    det = onset_idx[onset_idx < n_scan]
    if len(det) == 0:
        return None, 0, False
    t = np.sort((ref1 + det) / fs)
    for ti in t:
        n_c = int(((t >= ti) & (t <= ti + CORROB_WIN)).sum())
        if n_c >= MIN_CORROB:
            return float(ti), n_c, False
    return float(t[0]), 1, True

def is_sync(onset_idx, n_scan, fs):
    
    det = onset_idx[onset_idx < n_scan]
    if len(det) < 2 or len(det) / len(onset_idx) < SYNC_FRAC:
        return False
    t = det / fs
    return (np.percentile(t, 90) - np.percentile(t, 10)) <= SYNC_SPAN

def find_elec_onset(data_bq, er_full, fs, n, onset):
    
    flags = []
    ref0 = max(int((onset - 30.0) * fs), 0)
    ref1 = ref0 + int(REF_WIN * fs)
    if (ref1 - ref0) < MIN_REF * fs or (n - ref1) < 10 * fs:
        return dict(t_gamma=np.nan, t_er=np.nan, t_elec=np.nan, shift=np.nan,
                    onset_eff=onset, reanchored=False, flags='NO_REF',
                    n_corrob_g=0, n_corrob_e=0, sync_g=False, sync_e=False)
    scan0, scan1 = ref1, min(int((onset + 60.0) * fs), n)

    t_gamma = t_er = None
    n_cg = n_ce = 0
    sync_g = sync_e = False

    norm_scan, norm_ref = compute_hfer(data_bq[:, scan0:scan1], data_bq[:, ref0:ref1], fs)
    on_g = stage1_onset(norm_scan, norm_ref, fs)
    t_gamma, n_cg, single_g = corroborated_earliest(on_g, scan0, fs, scan1 - scan0)
    sync_g = is_sync(on_g, scan1 - scan0, fs)
    if single_g:
        flags.append('SINGLE_G')

    on_e = stage1_onset(er_full[:, scan0:scan1], er_full[:, ref0:ref1], fs)
    t_er, n_ce, single_e = corroborated_earliest(on_e, scan0, fs, scan1 - scan0)
    sync_e = is_sync(on_e, scan1 - scan0, fs)
    if single_e:
        flags.append('SINGLE_E')

    cands = [(t, 'g') for t in [t_gamma] if t is not None] + \
            [(t, 'e') for t in [t_er] if t is not None]
    if not cands:
        return dict(t_gamma=np.nan, t_er=np.nan, t_elec=np.nan, shift=np.nan,
                    onset_eff=onset, reanchored=False, flags='NO_DETECT',
                    n_corrob_g=0, n_corrob_e=0, sync_g=sync_g, sync_e=sync_e)
    t_elec, which = min(cands)
    flags.append('SRC_' + which.upper())
    if t_elec <= scan0 / fs + 0.5:
        flags.append('EDGE')
    if sync_g or sync_e:
        flags.append('SYNC')

    shift = t_elec - onset
    onset_eff, reanchored = onset, False
    if abs(shift) > REAN_TOL:
        b0 = max(int((t_elec - BASE_WIN[0]) * fs), 0)
        b1 = int((t_elec - BASE_WIN[1]) * fs)
        t0 = int((t_elec - TARGET_WIN[0]) * fs)
        t1 = min(int((t_elec + TARGET_WIN[1]) * fs), n)
        if (b1 - b0) >= MIN_BASE * fs and (t1 - t0) >= MIN_TARGET * fs:
            onset_eff, reanchored = t_elec, True
        else:
            flags.append('CANNOT_REANCHOR')
    return dict(t_gamma=t_gamma, t_er=t_er, t_elec=t_elec, shift=shift,
                onset_eff=onset_eff, reanchored=reanchored, flags=';'.join(flags),
                n_corrob_g=n_cg, n_corrob_e=n_ce, sync_g=sync_g, sync_e=sync_e)

def run_segment(fif, onset):
    
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    fs = float(raw.info['sfreq'])
    data = raw.get_data().astype(np.float64)
    ch_names = list(raw.ch_names)
    n = data.shape[1]

    data_bq = brainquake_filter(data, fs)
    er_full = er_series(data, fs)

    st1 = find_elec_onset(data_bq, er_full, fs, n, onset)
    onset_eff = st1['onset_eff']

    b0 = max(int((onset_eff - BASE_WIN[0]) * fs), 0)
    b1 = int((onset_eff - BASE_WIN[1]) * fs)
    t0 = int((onset_eff - TARGET_WIN[0]) * fs)
    t1 = min(int((onset_eff + TARGET_WIN[1]) * fs), n)
    if b1 - b0 < 5 * fs or t1 - t0 < 5 * fs:
        return None, st1

    norm_t, norm_b = compute_hfer(data_bq[:, t0:t1], data_bq[:, b0:b1], fs)
    n_t = norm_t.shape[1]

    out = {}
    thr0 = bq_thresholds(norm_b)
    on0 = np.where((norm_t > thr0[:, None]).any(axis=1),
                   np.argmax(norm_t > thr0[:, None], axis=1), n_t)
    out['bq20'] = bq_ei(norm_t, on0, fs, self_anchor=False)[0]
    on1, _ = sustained_onset(norm_t, norm_b, fs, mode='p99')
    out['p99'] = bq_ei(norm_t, on1, fs, self_anchor=False)[0]
    on1b, _ = sustained_onset(norm_t, norm_b, fs, mode='mad')
    out['mad'] = bq_ei(norm_t, on1b, fs, self_anchor=False)[0]
    out['self'] = bq_ei(norm_t, on1b, fs, self_anchor=True)[0]
    onsets = {'bq20': on0, 'p99': on1, 'mad': on1b, 'self': on1b}

    er_t, er_b = er_full[:, t0:t1], er_full[:, b0:b1]
    on3, _ = sustained_onset(er_t, er_b, fs, mode='mad')
    out['er'] = er_ei(er_t, on3, fs)
    onsets['er'] = on3

    t_on = {v: np.where(onsets[v] < n_t, (t0 + onsets[v]) / fs - onset_eff, np.nan)
            for v in out}
    return dict(ch_names=ch_names, ei=out, t_onset=t_on, fs=fs), st1

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = pd.read_csv(MAP_CSV, encoding='utf-8-sig')
    ch_map_norm = {p: g.set_index('channel') for p, g in cmap.groupby('patient')}
    pt_grade = cmap.drop_duplicates('patient').set_index('patient')['grade'].to_dict()
    pt_center = cmap.drop_duplicates('patient').set_index('patient')['center'].to_dict()
    soz_map = {}
    if os.path.isfile(SOZ_CSV):
        soz = pd.read_csv(SOZ_CSV, encoding='utf-8-sig')
        soz['ch_n'] = soz.channel.map(norm_ch)
        soz_map = {p: g.set_index('ch_n')['is_soz'] for p, g in soz.groupby('patient')}
        print(f"[SOZ] expert annotations loaded: {soz.patient.nunique()} patients")

    manifests = [pd.read_csv(mc, encoding='utf-8-sig')
                 for mc in sorted(glob.glob(os.path.join(ICTAL_ROOT, '_manifest_*.csv')))
                 if 'all' not in os.path.basename(mc)]
    man = pd.concat(manifests, ignore_index=True)
    man = man[man['status'].isin(['OK', 'SHORT'])].copy()
    print(f"segments to process: {len(man)} (patients {man['patient'].nunique()})")

    ch_rows, seg_rows, shift_rows = [], [], []
    t0 = time.time()
    for _, r in man.iterrows():
        pt, edf = r['patient'], r['edf']
        fif = os.path.join(ICTAL_ROOT, r['center'], pt, f"{pt}_{edf}_ictal90s.fif")
        if not os.path.isfile(fif):
            continue
        onset = float(r['onset_in_seg'])
        res, st1 = run_segment(fif, onset)

        shift_rows.append(dict(
            patient=pt, edf=edf, center=r['center'],
            onset_annot_sec=onset,
            t_gamma=st1['t_gamma'], t_er=st1['t_er'], t_elec=st1['t_elec'],
            shift=st1['shift'], reanchored=st1['reanchored'],
            flags=st1['flags'],
            n_corrob_gamma=st1['n_corrob_g'], n_corrob_er=st1['n_corrob_e'],
            sync_gamma=st1['sync_g'], sync_er=st1['sync_e'],
        ))
        if res is None:
            print(f"  [SKIP] {pt}/{edf}: insufficient window")
            continue
        chs = res['ch_names']
        dense = flags_for(pt, chs, ch_map_norm, 'is_dense')
        ez = flags_for(pt, chs, ch_map_norm, 'ez')
        src = flags_for(pt, chs, ch_map_norm, 'is_source')
        srr = flags_for(pt, chs, ch_map_norm, 'sr_rate')
        soz = np.array([soz_map.get(pt, {}).get(norm_ch(c), np.nan) for c in chs], float) \
            if soz_map else np.full(len(chs), np.nan)
        m = np.isfinite(dense)
        if m.sum() < 10:
            continue

        for variant in ['bq20', 'p99', 'mad', 'self', 'er']:
            ei = res['ei'][variant]
            ton = res['t_onset'][variant]
            det = np.isfinite(ton)
            topq = np.nanquantile(ei[m], 0.9)
            top = (ei >= topq) & (topq > 0)
            def _rate(flag):
                mm = m & top & np.isfinite(flag)
                return float(flag[mm].mean()) if mm.any() else np.nan
            iqr = (float(np.percentile(ton[det], 75) - np.percentile(ton[det], 25))
                   if det.sum() >= 4 else np.nan)
            seg_rows.append(dict(
                patient=pt, edf=edf, variant=variant,
                center=r['center'], grade=pt_grade.get(pt),
                onset_annot_sec=onset, shift=st1['shift'],
                reanchored=st1['reanchored'], anchor_flags=st1['flags'],
                n_ch=len(chs), n_matched=int(m.sum()),
                n_dense=int((dense[m] == 1).sum()),
                frac_detected=float(det.mean()), onset_iqr=iqr,
                earliest_vs_onset=float(np.nanmin(ton)) if det.any() else np.nan,
                P_dense_top10=_rate(dense), P_src_top10=_rate(src),
                P_ez_top10=_rate(ez), P_soz_top10=_rate(soz),
                top1_dense=float(dense[m][np.nanargmax(ei[m])]),
                top1_src=float(src[m][np.nanargmax(ei[m])]),
                base_dense=float(np.nanmean(dense[m])),
                base_src=float(np.nanmean(src[m])),
            ))
            for c, e, t, d, z, s, sr, oz in zip(chs, ei, ton, dense, ez, src, srr, soz):
                ch_rows.append(dict(patient=pt, edf=edf, channel=c, variant=variant,
                                    EI=e, t_onset=t, detected=float(np.isfinite(t)),
                                    is_dense=d, ez=z, is_source=s, is_soz=oz,
                                    sr_rate=sr))
        print(f"  {pt}/{edf}: shift={st1['shift']:+.1f}s "
              f"reanchor={st1['reanchored']} ({time.time()-t0:.0f}s)")

    ch_df = pd.DataFrame(ch_rows)
    seg_df = pd.DataFrame(seg_rows)
    shift_df = pd.DataFrame(shift_rows)
    for v, g in seg_df.groupby('variant'):
        for pt, gg in g.groupby('patient'):
            idx = gg.index
            seg_df.loc[idx, 'quality'] = (zs(gg.frac_detected) - zs(gg.onset_iqr)
                                          - zs(np.abs(gg.earliest_vs_onset.fillna(1e9))))
    ch_df.to_csv(os.path.join(OUT_DIR, 'part12_ei_v5_channel_level.csv'),
                 index=False, encoding='utf-8-sig')
    seg_df.to_csv(os.path.join(OUT_DIR, 'part12_ei_v5_segment_level.csv'),
                  index=False, encoding='utf-8-sig')
    shift_df.to_csv(os.path.join(OUT_DIR, 'part12_ei_v5_onset_shift.csv'),
                    index=False, encoding='utf-8-sig')

    from itertools import combinations
    pt_rows = []
    for pt, g in ch_df.groupby('patient'):
        row = dict(patient=pt, n_segments=g.edf.nunique(),
                   grade=pt_grade.get(pt), center=pt_center.get(pt))
        sh = shift_df[shift_df.patient == pt]
        if len(sh):
            row['n_reanchored'] = int(sh.reanchored.sum())
            row['shift_med'] = float(sh['shift'].median())
            row['shift_flags'] = ';'.join(sorted(set(
                f for fl in sh.flags.fillna('') for f in str(fl).split(';') if f)))
        for variant, gv in g.groupby('variant'):
            pfx = variant
            segs = {e: s.set_index('channel').EI for e, s in gv.groupby('edf')}
            rhos = []
            for e1, e2 in combinations(segs.keys(), 2):
                common = segs[e1].index.intersection(segs[e2].index)
                if len(common) < 10:
                    continue
                a, b = segs[e1][common].values, segs[e2][common].values
                if np.std(a) == 0 or np.std(b) == 0:
                    continue
                rhos.append(spearmanr(a, b).statistic)
            row[f'{pfx}_consistency'] = float(np.median(rhos)) if rhos else np.nan
            for label, name in [('ez', 'ez'), ('is_soz', 'soz')]:
                aucs = []
                for edf, gg in gv.dropna(subset=[label]).groupby('edf'):
                    lab = gg[label].values
                    if 0 < lab.sum() < len(lab):
                        aucs.append(roc_auc_score(lab, gg.EI.values))
                row[f'{pfx}_{name}_auc'] = float(np.mean(aucs)) if aucs else np.nan
            rhos2 = []
            for edf, gg in gv.dropna(subset=['sr_rate']).groupby('edf'):
                mm = np.isfinite(gg.EI.values) & np.isfinite(gg.sr_rate.values)
                if mm.sum() >= 10 and np.std(gg.sr_rate.values[mm]) > 0:
                    rhos2.append(spearmanr(gg.EI.values[mm], gg.sr_rate.values[mm]).statistic)
            row[f'{pfx}_rho_srrate'] = float(np.mean(rhos2)) if rhos2 else np.nan
            sv = seg_df[(seg_df.patient == pt) & (seg_df.variant == variant)]
            row[f'{pfx}_Pdense_med'] = float(sv.P_dense_top10.median()) if len(sv) else np.nan
            row[f'{pfx}_base_dense'] = float(sv.base_dense.median()) if len(sv) else np.nan
            row[f'{pfx}_fracdet_med'] = float(sv.frac_detected.median()) if len(sv) else np.nan
        pt_rows.append(row)
    pt_df = pd.DataFrame(pt_rows)
    pt_df.to_csv(os.path.join(OUT_DIR, 'part12_ei_v5_patient_level.csv'),
                 index=False, encoding='utf-8-sig')

    L = ['# Part12 v5.1 result summary (electrical onset re-anchored·bounded scan + five rulers)', '',
         '## 0. re-anchored audit (pre-registered rule: reference window[onset-30,-20]s, bounded scan[onset-20,+60]s, '
         'two detectors+sustained elevation≥2s+paired evidence, |shift|>5s re-anchored)', '',
         ' (v5.0 full-window free-scan version showed negative effect and wasdeprecated: anchor drifted before the annotation 60~110s  pre-ictal'
         'spike enhancement, detection rate and consistency both dropped, see script header v5.1 note)', '']
    if len(shift_df):
        rean = shift_df[shift_df.reanchored]
        L.append(f"- segmentstotal {len(shift_df)} (patients {shift_df.patient.nunique()}); "
                 f"re-anchored {len(rean)} segments (patients {rean.patient.nunique()} cases)")
        L.append(f"- |shift| median {shift_df['shift'].abs().median():.1f}s; "
                 f"|shift|>5s segments {(shift_df['shift'].abs() > 5).sum()}; "
                 f">10s segments {(shift_df['shift'].abs() > 10).sum()}")
        flag_stat = {}
        for fl in shift_df.flags.fillna(''):
            for f in str(fl).split(';'):
                if f:
                    flag_stat[f] = flag_stat.get(f, 0) + 1
        L.append(f"- flag statistics: {flag_stat}")
        if len(rean):
            L.append('- re-anchoredpatients: ' + ', '.join(
                f"{p}({int(g.reanchored.sum())}/{len(g)}segments,median shift{g['shift'].median():+.1f}s)"
                for p, g in shift_df[shift_df.reanchored].groupby('patient')))
    L += ['', '## 1. ruler reliability (cross-segmentconsistency)', '',
          '| ruler | median consistency | >=0.6 patients | median detection rate |', '|---|---|---|---|']
    for v in ['bq20', 'p99', 'mad', 'self', 'er']:
        c = pt_df[f'{v}_consistency'].dropna()
        fd = pt_df[f'{v}_fracdet_med'].dropna()
        if len(c):
            L.append(f"| {v} | {c.median():.3f} | {(c>=0.6).sum()}/{len(c)} | {fd.median():.3f} |")
        else:
            L.append(f"| {v} | no data | - | - |")
    L += ['', '## 3. validation (EZ=resected zonewide label, direction only; SOZ=expert annotation to be added for formal validation)']
    for v in ['bq20', 'p99', 'mad', 'self', 'er']:
        ez_ = pt_df[f'{v}_ez_auc'].dropna()
        soz_ = pt_df[f'{v}_soz_auc'].dropna()
        line = f"- {v}: EZ-AUC median {ez_.median():.3f}" if len(ez_) else f"- {v}: EZ-AUC no data"
        if len(soz_):
            line += f" | **SOZ-AUC median {soz_.median():.3f}, >0.5 patients {(soz_>0.5).sum()}/{len(soz_)}**"
        else:
            line += " | SOZ not annotated"
        L.append(line)
    with open(os.path.join(OUT_DIR, 'part12_ei_v5_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))
    print(f'\noutput → {OUT_DIR}')

if __name__ == '__main__':
    main()

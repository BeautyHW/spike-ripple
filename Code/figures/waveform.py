import os, sys
import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, r'<PROJECT_DIR>')
from _source_lib import load_sr_events, cluster_source

FIF = r'<DATA_DIR>\<PREPROCESSED_DIR>\<PATIENT>\<PATIENT>_SZ1_300s_raw.fif'
NPZ = r'<DATA_DIR>\last\<TIMESTAMPS_DIR>\<PATIENT>_sr.npz'
OUT_DIR = r'<OUTPUT_DIR>\_fig_output'

T0, TOL = 85.284, 0.03
RIPPLE_BAND = (80, 250)
PAD_S = 0.6
SR_COOC = 0.035

LANE_UV = 300.0
LANE_IN = 0.85

def norm_ch(name):
    return (str(name).replace("'", "").replace("\u2032", "")
            .replace("\u02b9", "").replace("`", "").replace(" ", "").strip())

def get_cluster(t0_target, tol):
    sr = load_sr_events(NPZ, anchor='spike_pos')
    res = cluster_source(sr['events'], win=0.05, lead=0.002)
    best = None
    for cl in res['clusters']:
        if abs(cl['t0'] - t0_target) <= tol:
            if best is None or abs(cl['t0'] - t0_target) < abs(best['t0'] - t0_target):
                best = cl
    if best is None:
        raise RuntimeError(f'no cluster found at t0≈{t0_target}  propagation cluster')
    return best

def pick_scale_uv(range_uv, choices=(500, 200, 100, 50, 20, 10)):
    
    for c in choices:
        if c <= range_uv * 0.40:
            return c
    return choices[-1]

def draw_scalebar(ax, x0_ms, y0, w_ms, h_uv, fontsize=8):
    
    ax.plot([x0_ms, x0_ms + w_ms], [y0, y0], color='black', lw=1.4,
            zorder=10, clip_on=False)
    ax.plot([x0_ms, x0_ms], [y0, y0 + h_uv], color='black', lw=1.4,
            zorder=10, clip_on=False)
    ax.text(x0_ms + w_ms / 2, y0 - h_uv * 0.06, f'{w_ms} ms',
            ha='center', va='top', fontsize=fontsize)
    ax.text(x0_ms + w_ms * 0.08, y0 + h_uv / 2, f'{h_uv} \u03bcV',
            ha='left', va='center', fontsize=fontsize)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    best = get_cluster(T0, TOL)
    channel_list = [str(ch) for _t, ch in best['members']]
    members_t = [float(t) for t, _ch in best['members']]
    n_ch = len(channel_list)
    t_src, t_last = members_t[0], members_t[-1]
    print(f'cluster: t0={t_src:.3f}s  source={best["source"]}  {n_ch}ch  '
          f'span={(t_last - t_src) * 1000:.1f}ms  lead={best["lead"] * 1000:.1f}ms')

    raw = mne.io.read_raw_fif(FIF, preload=False, verbose=False)
    sfreq = int(raw.info['sfreq'])
    fif_norm = [norm_ch(c) for c in raw.ch_names]
    idxs = [fif_norm.index(norm_ch(ch)) for ch in channel_list]
    raw_n = raw.copy().pick([raw.ch_names[i] for i in idxs]).load_data()
    data = (raw_n.get_data() * 1e6).astype(np.float32)
    del raw_n
    n_times = data.shape[1]
    print(f'sfreq={sfreq}Hz  data shape={data.shape}')

    peaks = []
    for k in range(n_ch):
        lo = max(0, int((members_t[k] - 0.08) * sfreq))
        hi = min(n_times, int((members_t[k] + 0.08) * sfreq))
        seg = data[k, lo:hi]
        j = int(np.argmax(np.abs(seg)))
        peaks.append(dict(t=(lo + j) / sfreq, v=float(seg[j]),
                          pp=float(np.ptp(seg))))

    order = np.argsort([p['t'] for p in peaks])
    t_ref = peaks[order[0]]['t']

    f0, f1 = t_src - 1.0, t_last + 1.2
    i0p = max(0, int((f0 - PAD_S) * sfreq))
    i1p = min(n_times, int((f1 + PAD_S) * sfreq))
    rip_all = mne.filter.filter_data(
        data[:, i0p:i1p].astype(np.float64), sfreq,
        RIPPLE_BAND[0], RIPPLE_BAND[1],
        method='fir', fir_design='firwin', phase='zero-double',
        n_jobs=1, verbose=False)

    print('\n[channel evaluation]  true peak time / spike pp / ripple pp (peak+/-60ms) / full-window ripple pp')
    stats = []
    for k in range(n_ch):
        tp = peaks[k]['t']
        a, b = int((tp - 0.06) * sfreq), int((tp + 0.06) * sfreq)
        rp = float(np.ptp(rip_all[k, a - i0p:b - i0p]))
        gw = float(np.ptp(rip_all[k, int((t_src - 0.5) * sfreq) - i0p:
                                  int((t_last + 0.5) * sfreq) - i0p]))
        stats.append((k, tp, peaks[k]['pp'], rp, gw))
        print(f'  {channel_list[k]:<8} t_peak={tp:.3f}s  spike_pp={peaks[k]["pp"]:7.1f}uV  '
              f'ripple@peak={rp:6.1f}uV  ripple full-window={gw:6.1f}uV')
    cand = [s for s in stats if s[2] >= 150.0]
    kA = max(cand if cand else stats, key=lambda s: s[3])[0]
    print(f'  -> Fig A channel: {channel_list[kA]}')

    w0, w1 = t_ref - 0.15, max(peaks[order[-1]]['t'] + 0.35, t_ref + 0.40)
    i0 = max(0, int(w0 * sfreq)); i1 = min(n_times, int(w1 * sfreq))
    times = np.arange(i0, i1, dtype=float) / sfreq
    x_ms_all = (times - t_ref) * 1000.0

    figB, axB = plt.subplots(figsize=(9.5, LANE_IN * n_ch + 1.1))
    for lane, k in enumerate(order):
        sig = data[k, i0:i1]
        mu = float(np.mean(sig))
        yl = n_ch - 1 - lane
        axB.plot(x_ms_all, (sig - mu) / LANE_UV + yl,
                 color='black', lw=1.1, zorder=3)
        tp, vp = peaks[k]['t'], peaks[k]['v']
        x_ev = (tp - t_ref) * 1000.0
        axB.plot([x_ev, x_ev], [yl - 0.82, yl + 0.82],
                 color='#C62828', lw=1.1, ls=(0, (4, 3)), zorder=2)
        axB.plot([x_ev], [(vp - mu) / LANE_UV + yl], marker='o',
                 mfc='white', mec='#C62828', mew=1.2, ms=5, zorder=5)
    axB.axvline(0, color='0.35', lw=0.9, ls=':', zorder=1)

    axB.set_ylim(-1.05, n_ch - 1 + 1.05)
    axB.set_yticks([n_ch - 1 - lane for lane in range(n_ch)])
    labels = []
    for lane, k in enumerate(order):
        dt = (peaks[k]['t'] - t_ref) * 1000.0
        lab = f"{channel_list[k]}  ({dt:+.0f} ms)"
        if channel_list[k] == best['source']:
            lab += ' \u2605'
        labels.append(lab)
    axB.set_yticklabels(labels, fontsize=8.5)

    axB.set_xlim((w0 - t_ref) * 1000, (w1 - t_ref) * 1000)
    axB.set_xlabel('Time from source peak (ms)', fontsize=10)
    axB.tick_params(labelsize=8)
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)

    x_sb = (w1 - t_ref) * 1000 - 130
    y_sb = -0.72
    axB.plot([x_sb, x_sb + 100], [y_sb, y_sb], 'k-', lw=1.4)
    axB.plot([x_sb, x_sb], [y_sb, y_sb + 1.0], 'k-', lw=1.4)
    axB.text(x_sb + 50, y_sb - 0.22, '100 ms', ha='center', fontsize=8)
    axB.text(x_sb + 10, y_sb + 0.5, f'{LANE_UV:.0f} \u03bcV',
             ha='left', va='center', fontsize=8)

    axB.set_title(f'Spike propagation \u2014 <PATIENT>, cluster t0={t_src:.3f}s  '
                  f'(source {best["source"]}, {n_ch} ch, span {(t_last - t_src) * 1000:.0f} ms)',
                  fontsize=10, loc='left')

    tpA = peaks[kA]['t']
    j0, j1 = int((tpA - 0.30) * sfreq), int((tpA + 0.30) * sfreq)
    xA = (np.arange(j0, j1, dtype=float) / sfreq - tpA) * 1000.0
    sigA = data[kA, j0:j1]
    muA = float(np.mean(sigA))
    ripA = rip_all[kA, j0 - i0p:j1 - i0p].astype(np.float64)
    ripA = ripA - float(np.mean(ripA))

    z = np.load(NPZ, allow_pickle=True)
    npz_ch = [str(c) for c in z['ch_names']]
    ci_npz = npz_ch.index(channel_list[kA])
    spA = np.asarray(z['spike_pos'][ci_npz], float)
    rsA = np.asarray(z['ripple_start'][ci_npz], float)
    reA = np.asarray(z['ripple_end'][ci_npz], float)
    s_sel = float(spA[np.argmin(np.abs(spA - members_t[kA]))]) if len(spA) else tpA
    rs_sel = re_sel = None
    if len(rsA) and len(spA):
        mid = (rsA + reA) / 2.0
        ok = np.abs(mid - s_sel) <= SR_COOC
        if ok.any():
            mid2 = np.where(ok, mid, 1e9)
            jj = int(np.argmin(np.abs(mid2 - s_sel)))
            rs_sel, re_sel = float(rsA[jj]), float(reA[jj])

    figA, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 5.8), sharex=True,
        gridspec_kw={'hspace': 0.12, 'height_ratios': [1.25, 0.75]})

    ax1.plot(xA, sigA - muA, color='black', lw=1.5)
    ax1.axvline(0, color='#2E7D32', ls='--', lw=1.2, alpha=0.9)
    ax1.set_ylabel('Spike (broadband)', fontsize=10)
    rng1 = float(np.max(np.abs(sigA - muA))) * 1.18
    ax1.set_ylim(-rng1, rng1)
    h1 = pick_scale_uv(2 * rng1)
    draw_scalebar(ax1, xA[0] + 20, -rng1 * 0.92, 100, h1)

    ax2.plot(xA, ripA, color='#00695C', lw=1.1)
    ax2.axvline(0, color='#2E7D32', ls='--', lw=1.2, alpha=0.9)
    if rs_sel is not None:
        ax2.axvspan((rs_sel - tpA) * 1000.0, (re_sel - tpA) * 1000.0,
                    color='#4DB6AC', alpha=0.22, zorder=0)
    ax2.set_ylabel('Ripple (80\u2013250 Hz, FIR)', fontsize=10)
    rng2 = float(np.max(np.abs(ripA))) * 1.25
    ax2.set_ylim(-rng2, rng2)
    h2 = pick_scale_uv(2 * rng2)
    draw_scalebar(ax2, xA[0] + 20, -rng2 * 0.92, 100, h2)

    ax2.set_xlim(xA[0], xA[-1])
    ax2.set_xlabel('Time from spike peak (ms)', fontsize=10)
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_yticks([])
    figA.suptitle(f'<PATIENT> \u2014 SR event, channel {channel_list[kA]}, '
                  f'peak at {tpA:.3f} s', fontsize=11, y=0.98)

    for tag, fig in [('figA_spike_ripple', figA), ('figB_propagation', figB)]:
        png = os.path.join(OUT_DIR, f'<PATIENT>_{tag}_85_284.png')
        fig.savefig(png, dpi=200)
        try:
            fig.savefig(png.replace('.png', '.pdf'))
        except PermissionError:
            print(f'  [warning] PDF occupied: {os.path.basename(png)}')
        plt.close(fig)
        print(f'saved {png}')

    print('\n===== manual check info (MNE around 85.2s) =====')
    print(f'patient: <PATIENT>   FIF: <PATIENT>_SZ1_300s_raw.fif')
    print(f'Fig B propagationcluster: t0={t_src:.3f}s, {n_ch}ch, clustersource={best["source"]}')
    for lane, k in enumerate(order):
        print(f'   lane{lane}  {channel_list[k]:<8} NPZ events={members_t[k]:.3f}s  '
              f'true peak={peaks[k]["t"]:.3f}s  \u0394={(peaks[k]["t"] - t_ref) * 1000:+6.1f}ms  '
              f'spike_pp={peaks[k]["pp"]:6.1f}uV')
    print(f'Fig A channel: {channel_list[kA]}  true peak={tpA:.3f}s  '
          f'(NPZ spike_pos={s_sel:.3f}s, diff{(tpA - s_sel) * 1000:+.0f}ms)')
    if rs_sel is not None:
        print(f'    co-occurring ripple: [{rs_sel:.3f}, {re_sel:.3f}]s  '
              f'dur={(re_sel - rs_sel) * 1000:.0f}ms  (NPZ measured)')

if __name__ == '__main__':
    main()

r
import mne, pandas as pd, numpy as np, os, glob, re, sys, warnings, openpyxl, datetime

warnings.filterwarnings("ignore")

LAN_DATA_DIR = r"<RAW_EDF_DIR>"
FU_DATA_DIR  = r"<RAW_EDF_DIR>"
LAN_LABEL_XLS = os.path.join(LAN_DATA_DIR, "label6.10*")
LAN_TIME_XLS  = os.path.join(LAN_DATA_DIR, "<TIME_LABEL_FILE>")
FU_LABEL_XLS  = r"<LABEL_FILE>"
OUT_ROOT = r"<DATA_DIR>\part12_ictal"

LINE_FREQ = 50
PRE_ONSET = 120.0
POST_ONSET = 60.0
MIN_PRE = 10.0
MIN_POST = 20.0
PAD = 5.0
TARGET_FS = 1000
LP_FREQ = 400.0

APOS = r"['`\u00b4\u2019]"

def canon_shaft(s):
    
    return s.strip().replace("POL ", "").translate({ord(c): ord("'") for c in "`´’"})

def extract_shaft(ch):
    if "Ref" in ch or "EKG" in ch.upper() or "ECG" in ch.upper(): return None
    m = re.match(rf"(.+[A-Za-z]{APOS}?)\s*(\d+)$", ch)
    if m: return canon_shaft(m.group(1))
    return None

def extract_contact(ch):
    if "Ref" in ch or "EKG" in ch.upper() or "ECG" in ch.upper(): return None
    m = re.match(rf"(.+[A-Za-z]{APOS}?)\s*(\d+)$", ch)
    if m: return int(m.group(2))
    return None

DUP_SUFFIX = re.compile(rf"^(POL [A-Za-z]+{APOS}?\s*\d+)-\d+$")
def mne_base(ch):
    m = DUP_SUFFIX.match(ch); return m.group(1) if m else ch

def dedupe_mne_renames(raw):
    seen, rename_map, drop_extra = set(), {}, []
    for ch in raw.ch_names:
        base = mne_base(ch)
        if base == ch: continue
        if base not in seen:
            seen.add(base); rename_map[ch] = base
        else:
            drop_extra.append(ch)
    if rename_map: raw.rename_channels(rename_map)
    if drop_extra: raw.drop_channels(drop_extra)

def parse_time_value(val):
    if val is None or pd.isna(val): return None
    if isinstance(val, datetime.time): return val.hour*3600 + val.minute*60 + val.second
    s = str(val).strip()
    if not s: return None
    m = re.match(r'(\d{1,2})\s*h\s*(\d{1,2})\s*m\s*(\d{1,2})', s)
    if m: return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))
    m = re.match(r'(\d{1,2}):(\d{2}):(\d{2})', s)
    if m: return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))
    digits = re.findall(r'\d+', s)
    if len(digits) >= 3: return int(digits[0])*3600 + int(digits[1])*60 + int(digits[2])
    return None

def _parse_time(time_str):
    for fmt in ['%Y/%m/%d %H:%M:%S %f', '%Y/%m/%d %H:%M:%S 000']:
        try: return datetime.datetime.strptime(time_str.strip(), fmt)
        except ValueError: pass
    return None

def parse_center_b_onset_csv(csv_path):
    
    try:
        df = pd.read_csv(csv_path, sep='\t', encoding='gbk', header=None,
                         names=['label', 'time', 'code'], skiprows=1)
    except Exception:
        return None
    rec_start_ts = onset_ts = None
    for _, row in df.iterrows():
        label = str(row['label']).strip()
        time_str = str(row['time']).strip()
        if 'REC START' in label:
            rec_start_ts = _parse_time(time_str)
        elif onset_ts is None and 'SPIKE' not in label and 'REC START' not in label and 'MARK' not in label:
            onset_ts = _parse_time(time_str)
    if rec_start_ts is None or onset_ts is None: return None
    return (onset_ts - rec_start_ts).total_seconds()

def parse_contact_set(text):
    
    if pd.isna(text) or str(text).strip() == "": return set()
    t = str(text)
    for ch in ["；", "，", "、", "\n", "\r", ","]: t = t.replace(ch, ";")
    s = set()
    for shaft, start, end in re.findall(rf"([A-Za-z]{APOS}?)\s*(\d+)\s*-\s*(\d+)", t):
        for n in range(int(start), int(end) + 1):
            _add_shaft_variants(s, shaft, n)
    for shaft, num in re.findall(rf"([A-Za-z]{APOS}?)\s*(\d+)", t):
        _add_shaft_variants(s, shaft, int(num))
    return s

def _add_shaft_variants(s, shaft, n):
    c = canon_shaft(shaft)
    s.add((c, n))
    stripped = c.rstrip("'")
    if stripped != c:
        s.add((stripped, n))

def match_contact(s, n, cset):
    return (s, n) in cset or (s, 0) in cset or \
           (s.rstrip("'`\u00b4\u2019"), n) in cset or (s.rstrip("'`\u00b4\u2019"), 0) in cset

def run_pipeline(edf_path, sub, center, edf_label, onset_sec, bad_set, ez_set):
    raw = mne.io.read_raw_edf(edf_path, preload=True, encoding="latin1", verbose=False)
    sfreq = raw.info["sfreq"]; total_dur = raw.n_times / sfreq
    dedupe_mne_renames(raw)
    n_orig = len(raw.ch_names)
    original_ch_names = raw.ch_names[:]
    print(f"    Raw: {n_orig}ch, {sfreq}Hz, {total_dur:.0f}s, onset={onset_sec:.0f}s")

    peri_start = max(onset_sec - PRE_ONSET, 0)
    peri_end = min(onset_sec + POST_ONSET, total_dur - 1/sfreq)
    pre_avail = onset_sec - peri_start
    post_avail = peri_end - onset_sec
    if peri_end - peri_start < 15:
        print(f"    FAIL: insufficient duration{peri_end-peri_start:.0f}s")
        return None, dict(status='FAIL_SHORT', pre=pre_avail, post=post_avail)
    status = 'OK' if (pre_avail >= MIN_PRE and post_avail >= MIN_POST) else 'SHORT'

    if center == 'center_a':
        bad_ch_list = [original_ch_names[idx-1] for idx in sorted(bad_set, reverse=True)
                       if 1 <= idx <= n_orig]
    else:
        bad_ch_list = []
        for ch in raw.ch_names:
            s = extract_shaft(ch); c = extract_contact(ch)
            if s is not None and c is not None and match_contact(s, c, bad_set):
                bad_ch_list.append(ch)
    if bad_ch_list:
        raw.drop_channels(bad_ch_list)
        print(f"    Removed bad channels: {len(bad_ch_list)}")

    if center == 'center_a':
        ch_to_orig_idx = {oname: i for i, oname in enumerate(original_ch_names, start=1)
                          if i not in bad_set}
        keep_ch, keep_orig_idx = [], []
        for ch in raw.ch_names:
            s = extract_shaft(ch); c = extract_contact(ch)
            if s is not None and c is not None:
                keep_ch.append(ch); keep_orig_idx.append(ch_to_orig_idx.get(ch, -1))
        raw.pick_channels(keep_ch)
    else:
        keep_ch = [ch for ch in raw.ch_names if extract_shaft(ch) is not None]
        raw.pick_channels(keep_ch)
        keep_orig_idx = None
    print(f"    Kept SEEG: {len(raw.ch_names)}ch")
    if len(raw.ch_names) < 5:
        return None, dict(status='FAIL_FEWCH', pre=pre_avail, post=post_avail)

    ct = raw.get_channel_types()
    ut = set(ct.values()) if isinstance(ct, dict) else set(ct)
    if ut != {"seeg"}: raw.set_channel_types({ch: "seeg" for ch in raw.ch_names})
    if sfreq > TARGET_FS:
        raw.filter(l_freq=None, h_freq=TARGET_FS/2.5, picks="all",
                   method="fir", fir_design="firwin", phase="zero-double", verbose=False)
        raw.resample(TARGET_FS, npad="auto", verbose=False); sfreq = TARGET_FS

    crop_start = max(peri_start - PAD, 0)
    crop_end = min(peri_end + PAD, total_dur - 1/sfreq)
    raw.crop(tmin=crop_start, tmax=crop_end)
    notch_fs = [f for f in [LINE_FREQ, LINE_FREQ*2, LINE_FREQ*3] if f < sfreq/2]
    if notch_fs: raw.notch_filter(freqs=notch_fs, picks="all", fir_design="firwin", verbose=False)
    raw.filter(l_freq=0.5, h_freq=min(LP_FREQ, sfreq/2-1), picks="all",
               method="fir", fir_design="firwin", verbose=False)

    offset = peri_start - crop_start
    t1 = offset
    t2 = min(offset + (peri_end - peri_start), raw.times[-1])
    if t2 - t1 < 15:
        return None, dict(status='FAIL_CROP', pre=pre_avail, post=post_avail)
    raw.crop(tmin=t1, tmax=t2 - 1/sfreq)
    seg_dur = raw.times[-1]
    onset_in_seg = onset_sec - peri_start

    groups = {}
    for ch in raw.ch_names:
        s = extract_shaft(ch); c = extract_contact(ch)
        if s is None or c is None: continue
        groups.setdefault(s, []).append((ch, c))
    pairs, bp_labels, src_map = [], [], {}
    for elec, contacts in groups.items():
        contacts.sort(key=lambda x: x[1])
        for i in range(len(contacts) - 1):
            c1, n1 = contacts[i]; c2, n2 = contacts[i + 1]
            pairs.append((c1, c2))
            bl = f"{elec}{n1}-{n2}"; bp_labels.append(bl); src_map[bl] = (c1, c2)
    if len(pairs) < 5:
        return None, dict(status='FAIL_BIPOLAR', pre=pre_avail, post=post_avail)
    data = raw.get_data()
    idx_of = {ch: i for i, ch in enumerate(raw.ch_names)}
    data_bp = np.zeros((len(pairs), raw.n_times))
    for i, (c1, c2) in enumerate(pairs):
        data_bp[i, :] = data[idx_of[c1], :] - data[idx_of[c2], :]
    raw_bp = mne.io.RawArray(data_bp, mne.create_info(bp_labels, sfreq, "seeg"))

    ez_labels = {}
    for bl, (c1, c2) in src_map.items():
        if center == 'center_a':
            if keep_orig_idx is not None:
                i1 = keep_ch.index(c1) if c1 in keep_ch else -1
                i2 = keep_ch.index(c2) if c2 in keep_ch else -1
                o1 = keep_orig_idx[i1] if i1 >= 0 else None
                o2 = keep_orig_idx[i2] if i2 >= 0 else None
                is_ez = (o1 is not None and o1 in ez_set) or (o2 is not None and o2 in ez_set)
            else:
                is_ez = False
        else:
            is_ez = False
            for chn in (c1, c2):
                s, n = extract_shaft(chn), extract_contact(chn)
                if s is not None and n is not None and match_contact(s, n, ez_set):
                    is_ez = True; break
        ez_labels[bl] = 1 if is_ez else 0

    sub_out = os.path.join(OUT_ROOT, center, sub); os.makedirs(sub_out, exist_ok=True)
    prefix = f"{sub}_{edf_label}"
    raw_bp.save(os.path.join(sub_out, f"{prefix}_ictal90s.fif"), overwrite=True, verbose=False)
    pd.DataFrame({"channel_name": list(ez_labels.keys()),
                  "EZ_NEZ": list(ez_labels.values())}).to_csv(
        os.path.join(sub_out, f"{prefix}_labels.csv"), index=False, encoding="utf-8-sig")

    meta = dict(status=status, pre=pre_avail, post=post_avail,
                onset_in_seg=onset_in_seg, seg_dur=seg_dur,
                n_bip=len(pairs), n_ez=int(sum(ez_labels.values())),
                sfreq=sfreq)
    return prefix, meta

args_patients = None

def _write_manifest(path, rows):
    
    out = pd.DataFrame(rows)
    mp = os.path.join(OUT_ROOT, os.path.basename(path))
    if os.path.isfile(mp):
        old = pd.read_csv(mp, encoding='utf-8-sig')
        keep = old[~old['patient'].isin(set(out['patient']))] if len(out) else old
        out = pd.concat([keep, out], ignore_index=True)
    out.to_csv(mp, index=False, encoding='utf-8-sig')

def run_center_a():
    label_xls = glob.glob(LAN_LABEL_XLS)[0]
    time_xls = glob.glob(LAN_TIME_XLS)[0]
    df_label = pd.read_excel(label_xls, engine="openpyxl")
    time_df = pd.read_excel(time_xls, engine="openpyxl")
    print(f"Label: {df_label.shape}, Time: {time_df.shape}")

    patient_data = {}
    for i in range(len(df_label)):
        pid = df_label.iloc[i, 1]
        if pd.isna(pid): continue
        pid = str(pid).strip()
        if pid == "nan" or len(pid) < 2: continue
        ez_nums = str(df_label.iloc[i, 4]).strip() if pd.notna(df_label.iloc[i, 4]) else ""
        bad_nums = str(df_label.iloc[i, 11]).strip() if pd.notna(df_label.iloc[i, 11]) else ""
        ez_set = set(int(x) for x in ez_nums.split() if x.strip().isdigit())
        bad_set = set(int(x) for x in bad_nums.split() if x.strip().isdigit())
        patient_data[pid] = {"ez_set": ez_set - bad_set, "bad_set": bad_set}

    def time_rows(sub):
        rows, cur = [], None
        for i in range(len(time_df)):
            nv = time_df.iloc[i, 1]
            if pd.notna(nv): cur = str(nv).strip()
            if cur == sub: rows.append(time_df.iloc[i])
        return rows

    rows_all = []
    for sub in (args_patients or sorted(patient_data.keys())):
        if sub not in patient_data:
            print(f"\n[MISS] {sub}: not in label table — skip")
            rows_all.append(dict(patient=sub, center='center_a', edf='', status='MISS_LABEL'))
            continue
        sz_files = sorted(glob.glob(os.path.join(LAN_DATA_DIR, sub, "SZ*.edf")))
        if not sz_files:
            print(f"\n[MISS] {sub}: no SZ*.edf")
            rows_all.append(dict(patient=sub, center='center_a', edf='', status='MISS_EDF'))
            continue
        trows = time_rows(sub)
        print(f"\n{'='*50}\n  {sub}: {len(sz_files)} EDFs, {len(trows)} time rows\n{'='*50}")
        for i, sz_path in enumerate(sz_files):
            edf_label = os.path.basename(sz_path).replace(".edf", "")
            onset_sec = None
            if i < len(trows):
                rec_start = parse_time_value(trows[i].iloc[10])
                onset_time = parse_time_value(trows[i].iloc[13])
                if rec_start is not None and onset_time is not None:
                    onset_sec = onset_time - rec_start
                    if onset_sec < -3600: onset_sec += 86400
                    elif onset_sec < 0: onset_sec += 86400
            if onset_sec is None:
                print(f"  {edf_label}: NO ONSET  - SKIP")
                rows_all.append(dict(patient=sub, center='center_a', edf=edf_label, status='NO_ONSET'))
                continue
            info = patient_data[sub]
            prefix, meta = run_pipeline(sz_path, sub, 'center_a', edf_label, onset_sec,
                                        info['bad_set'], info['ez_set'])
            row = dict(patient=sub, center='center_a', edf=edf_label,
                       onset_sec=round(onset_sec, 1))
            if prefix is None:
                row.update(meta); row['status'] = meta['status']
            else:
                row.update(meta)
            rows_all.append(row)
    _write_manifest(os.path.join(OUT_ROOT, "_manifest_center_a.csv"), rows_all)
    return rows_all

def run_center_b():
    df_label = pd.read_excel(FU_LABEL_XLS, engine="openpyxl")
    bad_map, ez_map = {}, {}
    for _, r in df_label.iterrows():
        pid = str(r["patient_ID"]).strip()
        bad_map[pid] = parse_contact_set(r["bad_channels"])
        ez_map[pid] = parse_contact_set(r["EZ"])
    missing = []
    if missing:
        print(f"[WARN] Center Blabel table missing: {missing}")

    rows_all = []
    for sub in (args_patients or sorted(bad_map.keys())):
        sz_dir = os.path.join(FU_DATA_DIR, sub, "SEEG SZ")
        if not os.path.isdir(sz_dir): sz_dir = os.path.join(FU_DATA_DIR, sub, "SEEG")
        if not os.path.isdir(sz_dir):
            print(f"\n[MISS] {sub}: no SEEG directory")
            rows_all.append(dict(patient=sub, center='center_b', edf='', status='MISS_DIR'))
            continue
        edfs = sorted(glob.glob(os.path.join(sz_dir, "*.edf")))
        if not edfs:
            print(f"\n[MISS] {sub}: no EDF")
            rows_all.append(dict(patient=sub, center='center_b', edf='', status='MISS_EDF'))
            continue
        print(f"\n{'='*50}\n  {sub}: {len(edfs)} EDFs\n{'='*50}")
        for edf_path in edfs:
            edf_label = os.path.basename(edf_path).replace(".edf", "")
            csv_path = edf_path.replace(".edf", ".csv")
            if not os.path.exists(csv_path):
                print(f"  {edf_label}: NO CSV  - SKIP")
                rows_all.append(dict(patient=sub, center='center_b', edf=edf_label, status='NO_CSV'))
                continue
            onset_sec = parse_center_b_onset_csv(csv_path)
            if onset_sec is None:
                print(f"  {edf_label}: NO ONSET  - SKIP")
                rows_all.append(dict(patient=sub, center='center_b', edf=edf_label, status='NO_ONSET'))
                continue
            prefix, meta = run_pipeline(edf_path, sub, 'center_b', edf_label, onset_sec,
                                        bad_map.get(sub, set()), ez_map.get(sub, set()))
            row = dict(patient=sub, center='center_b', edf=edf_label,
                       onset_sec=round(onset_sec, 1))
            row.update(meta if prefix is not None else meta)
            if prefix is None: row['status'] = meta['status']
            rows_all.append(row)
    _write_manifest(os.path.join(OUT_ROOT, "_manifest_center_b.csv"), rows_all)
    return rows_all

def main():
    global args_patients
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if len(sys.argv) > 2:
        args_patients = sys.argv[2:]
    os.makedirs(OUT_ROOT, exist_ok=True)
    if which in ('center_a', 'all'):
        run_center_a()
    if which in ('center_b', 'all'):
        run_center_b()

    parts = []
    for mc in ('_manifest_center_a.csv', '_manifest_center_b.csv'):
        mp = os.path.join(OUT_ROOT, mc)
        if os.path.isfile(mp):
            parts.append(pd.read_csv(mp, encoding='utf-8-sig'))
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(OUT_ROOT, "_manifest_all.csv"), index=False, encoding='utf-8-sig')
        print("\n" + "=" * 60)
        print("  Part12 preprocessing summary (merged)")
        print("=" * 60)
        print(df['status'].value_counts().to_string())
        ok = df[df['status'] == 'OK']
        print(f"\nOK segments: {len(ok)} (patients {ok['patient'].nunique()}/50)")
        pt_ok = set(ok['patient'])
        no_data = []
        print(f"patients without OK segments({len(no_data)}): {no_data}")
    print("Done.")

if __name__ == "__main__":
    main()

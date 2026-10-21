import mne, pandas as pd, numpy as np, os, glob, re, warnings
warnings.filterwarnings("ignore")

BASE_DIR = r"<DATA_DIR>"
OUT_DIR = os.path.join(BASE_DIR, "<PREPROCESSED_DIR>")
os.makedirs(OUT_DIR, exist_ok=True)

LINE_HARMS = [50, 100, 150, 200]
DURATION = 300.0
T_START = 0.0

LABEL_XLS = r"<LABEL_FILE>"
df_xl = pd.read_excel(LABEL_XLS, engine='openpyxl')

patient_info = {}
for i in range(len(df_xl)):
    name = str(df_xl.iloc[i, 1]).strip()

    ez_set = set()
    col3 = df_xl.iloc[i, 3]
    if pd.notna(col3):
        for ch in str(col3).split(chr(92)):
            ch = ch.strip()
            if ch: ez_set.add(ch)

    bad_global = set()
    if df_xl.shape[1] > 5:
        bs = str(df_xl.iloc[i, -1])
        if bs and bs != 'nan':
            for x in bs.split():
                if x.strip().isdigit():
                    bad_global.add(int(x.strip()))

    patient_info[name] = {'ez_monopolar': ez_set, 'bad_global': bad_global}

def extract_shaft_and_contact(ch):
    
    ch = ch.replace('POL ', '').strip()
    m = re.match(r"([A-Za-z]+[$]?['`´]?)\s*(\d+)$", ch)
    if m: return m.group(1), int(m.group(2))
    return None, None

def remove_line_noise_spectrum(data, sfreq):
    
    N = data.shape[1]
    freqs = np.fft.rfftfreq(N, 1/sfreq)
    bw = 0.8
    X = np.fft.rfft(data, axis=1)
    for i in range(data.shape[0]):
        for f0 in LINE_HARMS:
            if f0 > sfreq/2 - 1: continue
            mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
            idx = np.where(mask)[0]
            if len(idx) < 1: continue
            left, right = idx[0]-1, idx[-1]+1
            if left < 0 or right >= len(freqs): continue
            for k, j in enumerate(idx):
                t = (k+1)/(len(idx)+1)
                X[i,j] = (X[i,left].real*(1-t) + X[i,right].real*t) + \
                         1j*(X[i,left].imag*(1-t) + X[i,right].imag*t)
    return np.fft.irfft(X, n=N, axis=1)

def bipolar_name(shaft, c1, c2):
    
    return f"{shaft}{c1}-{c2}"

n_done, n_skip = 0, 0
patient_dirs = sorted([d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
    and d not in ('_preprocessed','<PREPROCESSED_DIR>','<PREPROCESSED_DIR>',
                  'features','code','ml_results','stat_results')
    and not d.startswith('.')])

for patient in patient_dirs:
    pdir = os.path.join(BASE_DIR, patient)
    edf_files = sorted(glob.glob(os.path.join(pdir, '*.edf')))
    if not edf_files:
        print(f'{patient}: NO EDF, skip')
        continue

    info = patient_info.get(patient, {'ez_monopolar': set(), 'bad_global': set()})
    ez_mono = info['ez_monopolar']
    bad_global = info['bad_global']

    for edf_path in edf_files:
        edf_name = os.path.basename(edf_path).replace('.edf','')
        seg_key = edf_name

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, encoding='latin1', verbose=False)
        except Exception as e:
            print(f'{patient}/{edf_name}: READ ERR {e}')
            continue

        sfreq = raw.info['sfreq']
        total_dur = raw.n_times / sfreq
        if DURATION > total_dur:
            print(f'{patient}/{edf_name}: too short ({total_dur:.0f}s)')
            continue

        raw.set_channel_types({ch: 'seeg' for ch in raw.ch_names})

        ch_names_orig = raw.ch_names[:]
        bad_names = set()
        if bad_global:
            for idx_1based in bad_global:
                if 1 <= idx_1based <= len(ch_names_orig):
                    bad_names.add(ch_names_orig[idx_1based - 1])

        if bad_names:
            keep = [ch for ch in ch_names_orig if ch not in bad_names]
            raw.pick_channels(keep)
            print(f'{patient}/{edf_name}: removed {len(bad_names)} bad channels ({sorted(bad_names)[:5]}...)')

        raw.crop(tmin=T_START, tmax=DURATION)

        data_raw = raw.get_data()
        data_clean = remove_line_noise_spectrum(data_raw, sfreq)
        raw._data = data_clean

        groups = {}
        for ch in raw.ch_names:
            shaft, contact = extract_shaft_and_contact(ch)
            if shaft and contact is not None:
                groups.setdefault(shaft, []).append((ch, contact))

        bip_names, bip_data = [], []
        for shaft, contacts in groups.items():
            contacts.sort(key=lambda x: x[1])
            for k in range(len(contacts)-1):
                ch1, c1 = contacts[k]
                ch2, c2 = contacts[k+1]
                i1 = raw.ch_names.index(ch1)
                i2 = raw.ch_names.index(ch2)
                diff = raw.get_data()[i2] - raw.get_data()[i1]
                bl = bipolar_name(shaft, c1, c2)
                bip_names.append(bl)
                bip_data.append(diff)

        if len(bip_data) < 2:
            print(f'{patient}/{edf_name}: only {len(bip_data)} bipolar channels')
            continue

        bip_data = np.array(bip_data)
        info_raw = mne.create_info(ch_names=bip_names, sfreq=sfreq, ch_types='seeg')
        raw_bip = mne.io.RawArray(bip_data, info_raw, verbose='error')

        ez_labels = []
        for bl in bip_names:
            m = re.match(r"^([A-Za-z]+[$]?['`´]?)(\d+)-(\d+)$", bl)
            if m:
                shaft, c1_str, c2_str = m.group(1), m.group(2), m.group(3)
                c1, c2 = int(c1_str), int(c2_str)
                src1 = f'POL {shaft}{c1}'
                src2 = f'POL {shaft}{c2}'
                ez_labels.append(1 if (src1 in ez_mono or src2 in ez_mono) else 0)
            else:
                ez_labels.append(0)

        sub_out = os.path.join(OUT_DIR, patient)
        os.makedirs(sub_out, exist_ok=True)
        fif_out = os.path.join(sub_out, f'{patient}_{edf_name}_300s_raw.fif')
        raw_bip.save(fif_out, overwrite=True, verbose='error')

        df_lbl = pd.DataFrame({'channel_name': bip_names, 'EZ_NEZ': ez_labels})
        csv_out = os.path.join(sub_out, f'{patient}_{edf_name}_labels.csv')
        df_lbl.to_csv(csv_out, index=False, encoding='utf-8-sig')

        print(f'{patient}/{edf_name}: {len(bip_names)}bip, {sum(ez_labels)}EZ, OK')
        n_done += 1

print(f'\nDone: {n_done} segments')

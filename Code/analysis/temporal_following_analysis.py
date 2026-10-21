import sys, os, glob, re, warnings, time, numpy as np, pandas as pd, mne
sys.path.insert(0, r'<DATA_DIR>\code')
warnings.filterwarnings('ignore')
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import maximum_filter1d
from hfo_detector import detect_hfo_channel
from _spike_tamilia_numba import parse_ch, are_adjacent

COOC_MS=20; SR_THRESH=2.0; START_MS=50; CHAIN_MS=10; MIN_CH=3
RIPPLE_BAND=(80,250); SPIKE_K=5.0; RIPPLE_NSTD=3.0
OUT=r'<PROJECT_DIR>'; os.makedirs(OUT,exist_ok=True)


def detect_spikes_2k(sig, fs, k=5.0):
    nyq=fs/2; b,a=butter(2,[10/nyq,60/nyq],'band'); sf=filtfilt(b,a,sig)
    env=np.abs(hilbert(sf)); med=np.median(env); mad=np.median(np.abs(env-med))
    th=med+k*mad*1.4826; lm=maximum_filter1d(env,size=13,mode='constant')
    peaks=(env==lm)&(env>th); sm=np.where(peaks)[0]; ts=sm/fs
    if len(ts)>1:
        kp=np.ones(len(ts),dtype=bool); la=0
        for i in range(1,len(ts)):
            if ts[i]-ts[la]<0.12: kp[i]=False
            else: la=i
        ts=ts[kp]
    return ts

print('='*70)
print('SR propagation timing — methodA(onset attribution) + methodB(region-levelmin latency)')
print('='*70)

results=[]; t0=time.time()

for dd in [r'<DATA_DIR>\<PREPROCESSED_DIR>',
           r'<DATA_DIR>\<PREPROCESSED_DIR>']:
    if not os.path.isdir(dd): continue
    for pt in sorted(os.listdir(dd)):
        ptd=os.path.join(dd,pt)
        if not os.path.isdir(ptd): continue
        fifs=sorted(glob.glob(os.path.join(ptd,'*_300s_raw.fif')))
        if not fifs: continue
        print('%s: '%pt,end='',flush=True); t1=time.time()
        try: raw=mne.io.read_raw_fif(fifs[0],preload=True,verbose='error')
        except: print('FIF_ERR'); continue
        fs=int(raw.info['sfreq']); data=raw.get_data(); ch=raw.ch_names; N=len(ch)
        dur_s=raw.n_times/fs

        lf=None
        for rep in ['_raw.fif','_300s_raw.fif']:
            cand=fifs[0].replace(rep,'_labels.csv')
            if os.path.isfile(cand): lf=cand; break
        if lf is None:
            alt=glob.glob(os.path.join(ptd,'*_labels.csv'))
            if alt: lf=alt[0]
        if lf is None:
            for root_d,_,files in os.walk(os.path.dirname(ptd)):
                for f in files:
                    if '_labels.csv' in f: lf=os.path.join(root_d,f); break
                if lf: break
        if lf is None: print('NO_LABEL'); continue
        dl=pd.read_csv(lf); col='channel_name' if 'channel_name' in dl.columns else 'channel'
        em=dict(zip(dl[col],dl['EZ_NEZ']))
        ez_clin=np.array([int(em.get(c,0)) for c in ch])

        adj=np.zeros((N,N),dtype=np.int8)
        for i in range(N):
            for j in range(N):
                if i!=j and are_adjacent(ch[i],ch[j]): adj[i,j]=1

        spikes=[np.zeros(0)]*N; ripples=[np.zeros(0)]*N
        for ci in range(N):
            if np.std(data[ci])<1e-15: continue
            spikes[ci]=detect_spikes_2k(data[ci],fs,SPIKE_K)
            re=detect_hfo_channel(data[ci],fs,RIPPLE_BAND,min_cycles=3,n_std=RIPPLE_NSTD)
            ripples[ci]=np.array([e['peak_time'] for e in re])

        sr_rate=np.zeros(N); sr_times_list=[]; sr_chs_list=[]
        win=COOC_MS/1000.0
        for ci in range(N):
            sts=spikes[ci]; rts=ripples[ci]
            if len(sts)==0 or len(rts)==0: continue
            cooc=0
            for rt in rts:
                if np.any(np.abs(sts-rt)<=win):
                    sr_times_list.append(rt); sr_chs_list.append(ci); cooc+=1
            sr_rate[ci]=cooc*60/dur_s

        if len(sr_times_list)==0:
            print('NO_SR'); continue

        sr_times=np.array(sr_times_list); sr_chs=np.array(sr_chs_list,dtype=np.int32)
        si=np.argsort(sr_times)
        sr_times=sr_times[si].astype(np.float64); sr_chs=sr_chs[si].astype(np.int32)

        sr_soz=set(ch[ci] for ci in range(N) if ez_clin[ci]==1 and sr_rate[ci]>=SR_THRESH)

        used=np.zeros(len(sr_times),dtype=np.int8)

        onset_is_srsoz=0; n_events=0

        on_ea_onfirst=0; on_ea_total=0

        for i in range(len(sr_times)):
            if used[i]: continue
            t_i=sr_times[i]; ch_i=sr_chs[i]
            seq_ch_set=np.zeros(N,dtype=np.int8); seq_ch_set[ch_i]=1
            j=i+1
            while j<len(sr_times) and sr_times[j]-t_i<=START_MS/1000.0:
                if used[j]: j+=1; continue
                ch_j=sr_chs[j]; is_adj=False
                for sc in range(N):
                    if seq_ch_set[sc] and adj[sc,ch_j]: is_adj=True; break
                if is_adj: seq_ch_set[ch_j]=1; used[j]=1
                j+=1
            changed=True
            while changed:
                changed=False; latest_t=0.0; end_j=j
                kk=j-1
                while kk>=i:
                    if used[kk]:
                        if sr_times[kk]>latest_t: latest_t=sr_times[kk]; end_j=kk
                    kk-=1
                j=end_j+1
                while j<len(sr_times) and sr_times[j]-latest_t<=CHAIN_MS/1000.0:
                    if used[j]: j+=1; continue
                    ch_j=sr_chs[j]; is_adj=False
                    for sc in range(N):
                        if seq_ch_set[sc] and adj[sc,ch_j]: is_adj=True; break
                    if is_adj: seq_ch_set[ch_j]=1; used[j]=1; changed=True
                    j+=1
            n_ch_in=seq_ch_set.sum()
            if n_ch_in<MIN_CH: continue
            n_events+=1
            onset_t=1e9
            kk=i
            while kk<len(sr_times):
                if used[kk] and seq_ch_set[sr_chs[kk]]:
                    if sr_times[kk]<onset_t: onset_t=sr_times[kk]
                kk+=1

            min_lat=1e9
            kk=i
            while kk<len(sr_times):
                if used[kk] and seq_ch_set[sr_chs[kk]]:
                    lat=sr_times[kk]-onset_t
                    if lat<min_lat: min_lat=lat
                kk+=1
            onset_chs=set()
            kk=i
            while kk<len(sr_times):
                if used[kk] and seq_ch_set[sr_chs[kk]]:
                    if (sr_times[kk]-onset_t)<=min_lat+1e-9:
                        onset_chs.add(sr_chs[kk])
                kk+=1
            if any(ch[cc] in sr_soz for cc in onset_chs):
                onset_is_srsoz+=1

            t_on=1e9; t_ea=1e9
            kk=i
            while kk<len(sr_times):
                if used[kk] and seq_ch_set[sr_chs[kk]]:
                    ch_k=sr_chs[kk]
                    lat_ms=(sr_times[kk]-onset_t)*1000.0
                    if ch[ch_k] in sr_soz:
                        if lat_ms<t_on: t_on=lat_ms
                    elif ez_clin[ch_k]==1:
                        if lat_ms<t_ea: t_ea=lat_ms
                kk+=1
            if t_on<1e9 and t_ea<1e9:
                on_ea_total+=1
                if t_on<t_ea: on_ea_onfirst+=1

        if n_events<3:
            print('FEW_EVENTS(%d)'%n_events); continue

        frac_a = onset_is_srsoz/n_events
        frac_b = on_ea_onfirst/on_ea_total if on_ea_total>0 else np.nan

        results.append({'patient':pt,'n_events':n_events,
            'A_onset_srsoz':onset_is_srsoz,'A_frac':frac_a,
            'B_on_ea_pairs':on_ea_total,'B_on_ea_first':on_ea_onfirst,'B_frac':frac_b})

        b_s='%.0f%%'%(100*frac_b) if not np.isnan(frac_b) else 'NA'
        print('%d events, A(onsetSR-SOZ)=%.0f%% B(region-levelon<ea)=%s (%.0fs)'%(
            n_events,100*frac_a,b_s,time.time()-t1))

df=pd.DataFrame(results)
out=os.path.join(OUT,'sr_temporal_final.csv')
df.to_csv(out,index=False,encoding='utf-8-sig')

print('\n'+'='*70)
print('SR propagation timing — two-method results')
print('='*70)
n=len(df)
print('validpatients: %d'%n)
print()
print('methodA (onset attribution): SR-SOZenrichmentat onset  ratio of casesmean=%.2f'%df.A_frac.mean())
print('  enrichment(>null hypothesis)patients: see v2 (92%%)')
print()
n_b=df.B_frac.notna().sum()
print('methodB (region-levelmin latency): Onset<Early ratio of casesmean=%.2f (null hypothesis0.5)'%df.B_frac.mean())
print('  Onset<Early patients: %d/%d (%.0f%%)'%((df.B_frac>0.5).sum(),n_b,100*(df.B_frac>0.5).sum()/n_b))
print()
for _,r in df.iterrows():
    b_s='%.0f%%'%(100*r.B_frac) if not np.isnan(r.B_frac) else 'NA'
    print('  %-16s: %3devents A=%.0f%% B(on<ea)=%s (%devents)'%(
        r.patient,r.n_events,100*r.A_frac,b_s,r.B_on_ea_pairs))
print('\nSaved: %s'%out)
print('Time: %.1fmin'%((time.time()-t0)/60))

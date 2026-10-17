# Spike Ripple as an Interictal Biomarker for Epileptogenic Zone Localization and Prognostic Prediction

Code for the manuscript *"Spike Ripple as an Interictal Biomarker for Epileptogenic Zone Localization and Prognostic Prediction in Drug-Resistant Epilepsy"*.

## Repository structure

```
spike-ripple/
├── preprocess/
│   ├── interictal_preprocess.py
│   └── ictal_preprocess.py
├── analysis/
│   ├── source_channel_determination.py
│   ├── high_rate_source_overlap.py
│   ├── source_channel_prognosis.py
│   ├── temporal_following_analysis.py
│   ├── granger_causality_net_flow.py
│   ├── granger_causality_net_flow_iiv.py
│   ├── epileptogenicity_index.py
│   └── monospike_polyspike.py
├── figures/
│   ├── waveform.py
│   ├── event_rate.py
│   ├── resection_ratio.py
│   ├── resected_frac.py
│   ├── overlap.py
│   ├── following.py
│   ├── gc.py
│   ├── roc.py
│   └── roc_subtype.py
└── README.md
```

## Data layout

```
<DATA_DIR>/
├── <patient>/
│   └── *.edf
└── <label_file>.xlsx
```

## Running

Install dependencies (`mne`, `numpy`, `pandas`, `matplotlib`, `scipy`, `openpyxl`, `scikit-learn`), edit the placeholder paths at the top of each script, then run the preprocessing scripts first.

## References

1. Shi W, Shaw D, Walsh KG, et al. Spike ripples localize the epileptogenic zone best: an international intracranial study. *Brain*. 2024;147(7):2496-2506.
2. Diamond JM, Withers CP, Chapeton JI, et al. Interictal discharges in the human brain are travelling waves arising from an epileptogenic source. *Brain*. 2023;146(5):1903-1915.
3. Withers CP, Diamond JM, Yang B, et al. Identifying sources of human interictal discharges with travelling wave and white matter propagation. *Brain*. 2023;146(12):5168-5181.
4. Matarrese MAG, Loppini A, Fabbri L, et al. Spike propagation mapping reveals effective connectivity and predicts surgical outcome in epilepsy. *Brain*. 2023;146(9):3898-3912.
5. Bartolomei F, Chauvel P, Wendling F. Epileptogenicity of brain structures in human temporal lobe epilepsy: a quantified study from intracerebral EEG. *Brain*. 2008;131(Pt 7):1818-1830.
6. Cai F, Wang K, Zhao T, et al. BrainQuake: an open-source Python toolbox for the stereoelectroencephalography spatiotemporal analysis. *Front Neuroinform*. 2022;15:773890.
7. Granger CWJ. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*. 1969;37(3):424-438.
8. Geweke J. Measurement of linear dependence and feedback between multiple time series. *J Am Stat Assoc*. 1982;77(378):304-313.

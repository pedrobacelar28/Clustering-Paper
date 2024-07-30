

It contain annotations about 6 different ECGs abnormalities:
- 1st degree AV block (1dAVb);
- right bundle branch block (RBBB);
- left bundle branch block (LBBB);
- sinus bradycardia (SB);
- atrial fibrillation (AF); and,
- sinus tachycardia (ST).




## Folder content:

- `ecg_selecionados.hdf5`:  The HDF5 file containing a single dataset named `tracings`. This dataset is a 
`(70, 4096, 12)` tensor. The first dimension correspond to the 70 different exams from different 
patients; the second dimension correspond to the 4096 signal samples; the third dimension to the 12
different leads of the ECG exams in the following order:
 `{DI, DII, DIII, AVL, AVF, AVR, V1, V2, V3, V4, V5, V6}`.

The signals are sampled at 400 Hz. Some signals originally have a duration of 
10 seconds (10 * 400 = 4000 samples) and others of 7 seconds (7 * 400 = 2800 samples).
In order to make them all have the same size (4096 samples) we fill them with zeros
on both sizes. For instance, for a 7 seconds ECG signal with 2800 samples we include 648
samples at the beginning and 648 samples at the end, yielding 4096 samples that are them saved
in the hdf5 dataset. All signal are represented as floating point numbers at the scale 1e-4V: so it should
be multiplied by 1000 in order to obtain the signals in V.

In python, one can read this file using the following sequence:
```python
import h5py
with h5py.File(args.tracings, "r") as f:
    x = np.array(f['tracings'])
```

- The file `attributes.csv` contain basic patient attributes: sex (M or F) and age. It
contain 827 lines (plus the header). The i-th tracing in `ecg_tracings.hdf5` correspond to the i-th line.
The  csv files  all have 6 columns `1dAVb, RBBB, LBBB, SB, AF, ST`
corresponding to weather the annotator have detect the abnormality in the ECG (`=1`) or not (`=0`).
  2. `gold_standard.csv` gold standard annotation for this test dataset. When the cardiologist 1 and cardiologist 2
  agree, the common diagnosis was considered as gold standard. In cases where there was any disagreement, a 
  third senior specialist, aware of the annotations from the other two, decided the diagnosis.

## Minhas Mudanças :
Eu separei 70 ECGs pra nossa vida ficar mais simples, 10 normais e 10 de cada doença, totalizando 70. Eles tao 
dispostos no arquivo nessa ordem:
- Normais 0-9;
- 1st degree AV block (1dAVb) 10-19;
- right bundle branch block (RBBB) 20-29;
- left bundle branch block (LBBB) 30-39;
- sinus bradycardia (SB) 40-49;
- atrial fibrillation (AF) 50-59;
- sinus tachycardia (ST) 60-69;

Perceba que depois do nome de cada doença coloquei os indices dos ecgs no novo arquivo, mas caso necessario
de referencia, os indices normais:

DataFrame com os índices dos primeiros 10 ECGs separados em colunas:
DataFrame transposto:
[[  0   2   3   4   5   6   7   8   9  10]
 [ 12  32  57  77  85  96 159 249 336 344]
 [106 241 255 289 298 313 330 365 367 379]
 [  1  28  58  99 104 141 185 217 251 253]
 [ 40  75  98 151 188 320 343 446 495 525]
 [120 170 259 348 355 368 408 501 548 564]
 [ 23  33  69 108 126 128 137 148 166 178]]
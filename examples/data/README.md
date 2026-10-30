# Empirical resting EEG example

`resting_eeg.npz` contains **real recorded EEG**, not a simulation: the first
30 seconds of channels C3, Cz and O1 from NeuroKit's eight-minute resting-state
example. Arrays are `data` (3 × 9000 float64 samples, volts), `channels`, and
`sfreq` (300 Hz). Load with `numpy.load(..., allow_pickle=False)`.

Source: [NeuroKit recording](https://github.com/neuropsychology/NeuroKit/blob/ff419d983568ef492eb8d229af643c0ef0100b32/data/eeg_resting_8min_300hz.fif).
Pinned revision: `ff419d983568ef492eb8d229af643c0ef0100b32`.
Source FIF SHA-256: `e479e9b1bd5ab06e1ff47dd2305e74995b2168b1586589b4c8c077b9da4d7afc`.
Its Git blob hash was independently checked against the pinned repository tree.

The upstream [preparation script](https://github.com/neuropsychology/NeuroKit/blob/ff419d983568ef492eb8d229af643c0ef0100b32/data/eeg_resting_8min.py)
documents conversion from a 3000 Hz EDF recording, cropping eight minutes after
an event onset, channel naming/montage handling and resampling to 300 Hz.
We only extract channels and the first 30 seconds; we apply no additional
filtering, resampling, re-referencing or artifact rejection. Acquisition details,
participant identity and eyes-open/eyes-closed condition are not documented by
that script; do not infer them. This small example demonstrates runtime, not
population-level EEG findings.

NeuroKit distributes the source file in its MIT-licensed repository. The full
upstream copyright and license notice is retained in
[LICENSE.NeuroKit.txt](LICENSE.NeuroKit.txt). This attribution does not impose a
license on the rest of this repository.

Please cite the source project when reusing this example:
Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
Schölzel, C., & Chen, S. H. A. (2021). *NeuroKit2: A Python toolbox for
neurophysiological signal processing*. Behavior Research Methods, 53, 1689–1696.
https://doi.org/10.3758/s13428-020-01516-y

To recreate the excerpt (downloads approximately 70 MiB over verified HTTPS):

```sh
python examples/prepare_resting_eeg.py
```

The script verifies the pinned SHA-256 before reading the recording with MNE.
For an existing local copy, use `--source /path/to/eeg_resting_8min_300hz.fif`.

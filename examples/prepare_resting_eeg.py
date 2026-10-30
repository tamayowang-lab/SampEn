"""Download the pinned NeuroKit recording and extract three EEG channels."""
import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

import mne
import numpy as np

COMMIT = 'ff419d983568ef492eb8d229af643c0ef0100b32'
URL = f'https://raw.githubusercontent.com/neuropsychology/NeuroKit/{COMMIT}/data/eeg_resting_8min_300hz.fif'
SHA256 = 'e479e9b1bd5ab06e1ff47dd2305e74995b2168b1586589b4c8c077b9da4d7afc'
ROOT = Path(__file__).resolve().parents[1]


def prepare(source, output):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Source recording SHA-256 mismatch')
    raw = mne.io.read_raw_fif(source, preload=False, verbose='ERROR')
    channels = ['C3', 'Cz', 'O1']
    # First 30 seconds, no added filtering, re-referencing, or resampling.
    n = int(30 * raw.info['sfreq'])
    data = raw.get_data(picks=channels, start=0, stop=n)
    assert data.shape == (3, 9000) and np.all(np.isfinite(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, data=data, channels=np.array(channels),
                        sfreq=np.array(raw.info['sfreq']))
    print(f'Wrote {output}: {data.shape}, {raw.info["sfreq"]} Hz, volts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Previously downloaded FIF file')
    parser.add_argument('--output', type=Path, default=ROOT/'examples/data/resting_eeg.npz')
    args = parser.parse_args()
    if args.source:
        prepare(args.source, args.output)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'eeg_resting_8min_300hz.fif'
            urllib.request.urlretrieve(URL, source)
            prepare(source, args.output)

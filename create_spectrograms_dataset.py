import pandas as pd
import argparse
from pathlib import Path
import runpy
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from tqdm import tqdm

def load_and_trim(path, sr, top_db):
    """Load audio and trim leading/trailing silence."""
    audio, _ = librosa.load(path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio


def compute_spectrogram(audio, sr, n_fft, hop_length, n_mels):
    """Return Mel spectrogram"""
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        n_mels=n_mels)
    
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db


def normalize(spectrogram):
    """Per-sample zero-mean, unit-variance normalization."""
    mean = spectrogram.mean()
    std  = spectrogram.std() if spectrogram.std() > 0 else 1.0
    return (spectrogram - mean) / std


def save_spectrogram(spectrogram, output_path):
    """Save `spec` either as .npy or as a single‐channel grayscale PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # create a grayscale image
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    librosa.display.specshow(
        data=spectrogram,
        ax=ax,
        cmap='gray',
        x_axis=None,
        y_axis=None
    )
    ax.set_axis_off()

    # save as PNG
    fig.savefig(
        output_path.with_suffix('.png'),
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)


def process_file(row: pd.Series, settings):
    """Create spectrograms for all possible segments of one audio file.

    Parameters
    ----------
    row : pd.Series
        Row from ``all_speakers.csv`` describing one audio file.
    settings : argparse.Namespace
        CLI arguments.

    Returns
    -------
    list[dict]
        A list of dictionaries representing rows for the output CSV.
    """

    relative_path = Path(row["filename"])
    full_path = settings.input_dir / relative_path

    audio = load_and_trim(full_path, settings.sr, settings.top_db)

    segment_len = int(settings.duration * settings.sr)
    max_start = len(audio) - segment_len
    entries = []

    for start in range(0, max_start + 1, segment_len):
        end = start + segment_len
        if end > len(audio):
            break
        segment = audio[start:end]
        spectrogram = compute_spectrogram(
            segment,
            settings.sr,
            settings.n_fft,
            settings.hop_length,
            settings.n_mels,
        )
        if settings.normalize:
            spectrogram = normalize(spectrogram)

        out_rel = relative_path.with_name(
            f"{relative_path.stem}_{int(start / settings.sr)}_{int((start / settings.sr) + settings.duration)}{relative_path.suffix}"
        )
        out_full = settings.output_dir / out_rel

        save_spectrogram(spectrogram, out_full)

        row_copy = row.copy()
        row_copy["filename"] = str(out_full.with_suffix(".png"))
        entries.append(row_copy.to_dict())

    return entries




def main():
    # parser for scripts input arguments
    parser = argparse.ArgumentParser(
        description="Preprocess Voices_devkit to spectrograms"
    )
    parser.add_argument('--input-dir',   type=Path, default='VOiCES_devkit')
    parser.add_argument('--output-dir',  type=Path, default='VOiCES_devkit_spectrograms')
    parser.add_argument('--sr',          type=int,   default=16000)
    parser.add_argument('--duration',    type=float, default=5.0,
                   help="fixed clip length in seconds")
    parser.add_argument('--top-db',      type=int,   default=20,
                   help="librosa trim top_db")
    parser.add_argument('--n_mels',      type=int,   default=64)
    parser.add_argument('--n_fft',       type=int,   default=400)
    parser.add_argument('--hop_length',  type=int,   default=160)
    parser.add_argument('--normalize',   action='store_true')
    settings = parser.parse_args()

    all_speakers_path = Path("all_speakers.csv")

    if all_speakers_path.exists():
        dataframe_all = pd.read_csv("all_speakers.csv")
    else:
        runpy.run_path("create_speakers_csv.py")
        dataframe_all = pd.read_csv("all_speakers.csv")

    records = []
    for _, row in tqdm(dataframe_all.iterrows(), total=len(dataframe_all), desc="Processing files"):
        records.extend(process_file(row, settings))

    output_df = pd.DataFrame(records)
    output_df.to_csv("VOiCES_devkit_spectrograms.csv", index=False)
    


if __name__ == '__main__':
    main()

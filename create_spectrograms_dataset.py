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
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

def get_audio_for_spectrogram_generation(path, sr, duration, top_db):
    """Load audio, trim leading/trailing silence, pad or cut to `duration` seconds."""
    audio, _ = librosa.load(path, sr=sr)

    # 1) remove silence
    audio, _ = librosa.effects.trim(audio, top_db=top_db)

    # 2) trim to full slices of duration
    trimmed_duration = librosa.get_duration(y=audio, sr=sr)
    whole_slices_count = int(trimmed_duration/duration)
    target_len = int(whole_slices_count * sr * duration)
   
    audio = audio[:target_len]

    return audio, whole_slices_count


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


def save_spectrogram(spectrogram, output_path, start_second, end_second):
    """Save `spec` either as .npy or as a single‐channel grayscale PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # append slice info to filename
    if start_second is not None and end_second is not None:
        suffix = f"_slice_{int(start_second)}_{int(end_second)}.png"
        output_path = output_path.with_name(output_path.stem + suffix)
    
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
        output_path,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)


def process_file(path: Path, settings):
    """Run full pipeline for one audio file, plus optional augmentations."""
    duration = settings.duration
    sampling_rate = settings.sr
    top_db = settings.top_db
    slice_length = duration*sampling_rate
    n_fft = settings.n_fft
    hop_length = settings.hop_length
    n_mels = settings.n_mels

    audio, slice_count = get_audio_for_spectrogram_generation(path, sampling_rate, duration, top_db)
    
    # original
    for index in range(0, slice_count):
        start = int(index * slice_length)
        end = int((index + 1) * slice_length)
        audio_slice = audio[start:end]
        spectrogram = compute_spectrogram(audio_slice, sampling_rate,n_fft,
                                           hop_length, n_mels
                                )
        if settings.normalize:
            spectrogram = normalize(spectrogram)

        # save
        # Original path
        original_path = path

        # Replacing VOiCES_devkit with VOiCES_devkit_spectrograms
        new_parts = [part if part != settings.input_dir.parts[0] else settings.output_dir.parts[0] for part in original_path.parts]

        output_path = Path(*new_parts)

        start_second = index*duration
        end_second = (index+1)*duration
         
        save_spectrogram(spectrogram, output_path, start_second, end_second)


def wrapped_file_process(path, settings):
    fullpath = settings.input_dir / path
    process_file(fullpath, settings)


def main():
    # parser for scripts input arguments
    parser = argparse.ArgumentParser(
        description="Preprocess Voices_devkit to spectrograms"
    )
    parser.add_argument('--input-dir',   type=Path, default='VOiCES_devkit')
    parser.add_argument('--output-dir',  type=Path, default='VOiCES_devkit_spectrograms')
    parser.add_argument('--sr',          type=int,   default=16000)
    parser.add_argument('--duration',    type=float, default=3.0,
                   help="fixed clip length in seconds")
    parser.add_argument('--top-db',      type=int,   default=15,
                   help="librosa trim top_db")
    parser.add_argument('--n_mels',      type=int,   default=128)
    parser.add_argument('--n_fft',       type=int,   default=1024)
    parser.add_argument('--hop_length',  type=int,   default=160)
    parser.add_argument('--normalize',   action='store_true')
    settings = parser.parse_args()

    all_speakers_path = Path("all_speakers.csv")

    if all_speakers_path.exists():
        dataframe_all = pd.read_csv("all_speakers.csv")
    else:
        runpy.run_path("create_speakers_csv.py")
        dataframe_all = pd.read_csv("all_speakers.csv")

    paths_distant = dataframe_all["filename"].tolist()
    paths_source = dataframe_all["source"].unique().tolist()

    paths = paths_distant + paths_source

    with ProcessPoolExecutor(max_workers=30) as executor:  # adjust num of threads
        futures = [executor.submit(wrapped_file_process, path, settings) for path in paths]
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            pass
    


if __name__ == '__main__':
    main()

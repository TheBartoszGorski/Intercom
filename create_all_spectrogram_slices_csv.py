import pandas as pd
from pathlib import Path
import argparse
import librosa
from tqdm import tqdm

def calculate_slices(audio_path, sr, duration, top_db):
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        total_duration = librosa.get_duration(y=y_trimmed, sr=sr)
        slice_count = int(total_duration // duration)
        return slice_count
    except Exception as e:
        print(f"⚠️ Error loading {audio_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="all_speakers.csv")
    parser.add_argument("--input-dir", type=str, default="VOiCES_devkit")
    parser.add_argument("--output-dir", type=str, default="VOiCES_devkit_spectrograms")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--top-db", type=int, default=20)
    parser.add_argument("--output-csv", type=str, default="all_spectrogram_slices.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating slice entries"):
        relative_path = row['filename']
        audio_path = Path(args.input_dir) / relative_path

        if not audio_path.exists():
            print(f"⚠️ File not found: {audio_path}")
            continue

        slice_count = calculate_slices(audio_path, args.sr, args.duration, args.top_db)

        for i in range(slice_count):
            start = int(i * args.duration)
            end = int((i + 1) * args.duration)

            output_file = Path(args.output_dir) / relative_path
            output_file = output_file.with_suffix("")  # remove .wav
            output_file = output_file.with_name(f"{output_file.stem}_slice_{start}_{end}.png")
            record = row.copy()
            record["filename"] = str(output_file)
            record["slice_start"] = start
            record["slice_end"] = end
            records.append(record)

    new_df = pd.DataFrame(records)
    new_df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved {len(new_df)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
from pathlib import Path

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from myCNNs import CNN1, CNN2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Voices_devkit to spectrograms"
    )
    parser.add_argument('--dataset-csv-dir', type=Path, default='all_spectrograms_classified.csv')
    parser.add_argument('--speaker-id', type=int, default=1052)
    parser.add_argument("--CNN-model", type=str, default='CNN2')
    settings = parser.parse_args()
    model_name = settings.CNN_mode

    # Define the same model architecture
    if model_name == 'CNN1':
        model = CNN1().to(device)
    elif model_name == 'CNN2':
        model = CNN2().to(device)
    else:
        print("Incorrect model type!")
        return

    # Load the saved weights
    model.load_state_dict(torch.load('best_model.pth'))

    # Set to evaluation mode (if you're using it for inference)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    # Load dataset
    df = pd.read_csv(settings.dataset_csv_dir)

    # Filter entries for the selected speaker
    speaker_df = df[df['speaker'] == settings.speaker_id]

    if speaker_df.empty:
        print("No spectrograms found for this speaker ID.")
        return

    # List files
    prefix = "VOiCES_devkit_spectrograms/distant-16k/speech/train/"
    print(f"\nFound {len(speaker_df)} spectrogram(s) for speaker {settings.speaker_id}:")
    for i, row in speaker_df.reset_index(drop=True).iterrows():
        short_name = row['filename'].replace(prefix, "")
        print(f"[{i}] {short_name}")

    # Ask user to choose
    index = int(input("\nEnter index of spectrogram to evaluate: "))
    if index < 0 or index >= len(speaker_df):
        print("Invalid index.")
        return

    # Load chosen spectrogram
    filename = speaker_df.reset_index(drop=True).iloc[index]['filename']
    image = Image.open(filename).convert('L')
    spectrogram = transform(image).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        output = model(spectrogram)
        probability = torch.sigmoid(output).item() 
        print(f"Probability of being allowed: {probability:.4f}")

        if output.item() >= 0.5:
            print("Prediction: allowed")
        else:
            print("Prediction: NOT allowed")

if __name__ == "__main__":
    main()
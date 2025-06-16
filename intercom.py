import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from myCNNs import CNN1, CNN2
from create_spectrograms_dataset import compute_spectrogram, load_and_trim, normalize
import os

sr = 16000
duration = 5.0
top_db = 20
n_fft = 400
hop_length = 160
n_mels = 64
normalize_flag = True
model_name = 'CNN1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleSpectrogramDataset(Dataset):
    def __init__(self, spectrogram_tensor, label):
        """
        spectrogram_tensor: torch.Tensor of shape [1, mel_bins, time]
        label: int or float (0 or 1)
        """
        self.tensor = spectrogram_tensor
        self.label = torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.tensor, self.label

# --- RECORD AND CONVERT TO SPECTROGRAM ---
def record_to_tensor():
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("Recording complete.")
    audio = audio.flatten()

    audio = load_and_trim(audio, sr, duration, top_db)
    spec_db = compute_spectrogram(audio, sr, n_fft, hop_length, n_mels, spec_type='mel')

    if normalize_flag:
        spec_db = normalize(spec_db)

    return torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

# --- MAIN ---
def main():
    if model_name == 'CNN1':
        model = CNN1().to(device)
    elif model_name == 'CNN2':
        model = CNN2().to(device)
    else:
        print("Incorrect model type!")
        return

    model_path = "fine_tuned_model.pth" if os.path.exists("fine_tuned_model.pth") else f"{model_name}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    while True:
        mode = input("Mode [verify/register/exit]: ").strip().lower()
        if mode == "exit":
            break

        spec_tensor = record_to_tensor()

        if mode == "register":
            print("Adding yourself as an allowed user...")
            dataset = SingleSpectrogramDataset(spec_tensor, label=1)
            loader = DataLoader(dataset, batch_size=1)

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(3):
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            torch.save(model.state_dict(), "fine_tuned_model.pth")
            print("✅ You have been registered.")

        elif mode == "verify":
            print("Verifying identity...")
            model.eval()
            with torch.no_grad():
                output = model(spec_tensor.to(device))
                prob = torch.sigmoid(output).item()
                print(f"Access probability: {prob:.3f}")
                print("✅ Access granted." if prob >= 0.5 else "❌ Access denied.")

        else:
            print("Invalid mode. Please type 'verify', 'register', or 'exit'.")

if __name__ == "__main__":
    main()
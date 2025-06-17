from pathlib import Path

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
from tqdm import tqdm
from myCNNs import CNN1, CNN2
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_all_spectrograms_classified(path, allowed_men_count, allowed_women_count):
    # Load the dataset
    df = pd.read_csv(path)

    # Get unique speakers by gender
    female_speakers = df[df['gender'] == 'F']['speaker'].unique()
    male_speakers = df[df['gender'] == 'M']['speaker'].unique()

    # Randomly pick allowed_women_count female and allowed_men_count male speakers
    np.random.seed(42)  # for reproducibility
    allowed_females = np.random.choice(female_speakers, allowed_women_count, replace=False)
    allowed_males = np.random.choice(male_speakers, allowed_men_count, replace=False)

    # Combine into a set of allowed speakers
    allowed_speakers = set(allowed_females.tolist() + allowed_males.tolist())

    # Add "allowed" column
    df['allowed'] = df['speaker'].apply(lambda x: 1 if x in allowed_speakers else 0)

    return df

def create_balanced_spectrograms_classified(path, allowed_men_count, allowed_women_count):
    # Load the dataset
    df = pd.read_csv(path)

    # Filter only files from 'none' noise folder
    df = df[df['filename'].str.contains('none')].copy()

    # Get unique speakers by gender
    female_speakers = df[df['gender'] == 'F']['speaker'].unique()
    male_speakers = df[df['gender'] == 'M']['speaker'].unique()

    # Fix total to 30 speakers: 15 from each gender
    np.random.seed(42)
    selected_females = np.random.choice(female_speakers, 25, replace=False)
    selected_males = np.random.choice(male_speakers, 25, replace=False)

    selected_speakers = set(selected_females.tolist() + selected_males.tolist())
    df = df[df['speaker'].isin(selected_speakers)].copy()

    # Mark a subset of these as 'allowed'
    allowed_females = np.random.choice(selected_females, allowed_women_count, replace=False)
    allowed_males = np.random.choice(selected_males, allowed_men_count, replace=False)
    allowed_speakers = set(allowed_females.tolist() + allowed_males.tolist())

    df['allowed'] = df['speaker'].apply(lambda x: 1 if x in allowed_speakers else 0)

    return df

def create_noiseless_spectrograms_classified(path, allowed_men_count, allowed_women_count):
    # Load the dataset
    df = pd.read_csv(path)

    # Filter only files from 'none' noise folder
    df = df[df['filename'].str.contains('none')].copy()

    # Group speakers by gender
    men = df[df['gender'] == 'M']['speaker'].unique().tolist()
    women = df[df['gender'] == 'F']['speaker'].unique().tolist()

    # Shuffle and sample allowed speakers
    allowed_men = random.sample(men, allowed_men_count)
    allowed_women = random.sample(women, allowed_women_count)
    allowed_speakers = set(allowed_men + allowed_women)

    # Label samples: 1 = allowed, 0 = not allowed
    df['allowed'] = df['speaker'].apply(lambda x: 1 if x in allowed_speakers else 0)

    return df

def create_balanced_noiseless_spectrograms_classified(path, allowed_men_count, allowed_women_count):
    # Load the dataset
    df = pd.read_csv(path)

    # Filter only files from 'none' noise folder
    df = df[df['filename'].str.contains('none')].copy()

    # Get unique speakers by gender
    female_speakers = df[df['gender'] == 'F']['speaker'].unique()
    male_speakers = df[df['gender'] == 'M']['speaker'].unique()

    # Fix total to 30 speakers: 15 from each gender
    np.random.seed(42)
    selected_females = np.random.choice(female_speakers, 10, replace=False)
    selected_males = np.random.choice(male_speakers, 10, replace=False)

    selected_speakers = set(selected_females.tolist() + selected_males.tolist())
    df = df[df['speaker'].isin(selected_speakers)].copy()

    # Mark a subset of these as 'allowed'
    allowed_females = np.random.choice(selected_females, allowed_women_count, replace=False)
    allowed_males = np.random.choice(selected_males, allowed_men_count, replace=False)
    allowed_speakers = set(allowed_females.tolist() + allowed_males.tolist())

    df['allowed'] = df['speaker'].apply(lambda x: 1 if x in allowed_speakers else 0)

    return df

def generate_dataframes(df):
    # Get unique transcripts
    unique_transcripts = df['transcript'].unique()

    # Shuffle and split transcript list into train, validate, test (10:1:1 ratio)
    np.random.shuffle(unique_transcripts)

    n_total = len(unique_transcripts)
    n_validate = n_total // 12
    n_test = n_validate
    n_train = n_total - n_validate - n_test

    train_transcripts = unique_transcripts[:n_train]
    validate_transcripts = unique_transcripts[n_train:n_train + n_validate]
    test_transcripts = unique_transcripts[n_train + n_validate:]

    # Create subsets based on transcript split
    train_df = df[df['transcript'].isin(train_transcripts)].copy()
    validate_df = df[df['transcript'].isin(validate_transcripts)].copy()
    test_df = df[df['transcript'].isin(test_transcripts)].copy()

    # Ensure test and validate have at least one entry from each allowed speaker
    allowed_speakers = df[df['allowed'] == 1]['speaker'].unique()

    def ensure_coverage(df_subset):
        speakers_present = df_subset['speaker'].unique()
        missing_speakers = set(allowed_speakers) - set(speakers_present)
        for speaker in missing_speakers:
            speaker_entries = df[(df['speaker'] == speaker) & (df['transcript'].isin(train_transcripts))]
            if not speaker_entries.empty:
                chosen_row = speaker_entries.sample(1, random_state=42)
                df_subset = pd.concat([df_subset, chosen_row])
        return df_subset

    # Apply the corrected coverage logic
    validate_df = ensure_coverage(validate_df)
    test_df = ensure_coverage(test_df)

    return train_df, validate_df, test_df


class SpectrogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['filename']

        image = Image.open(path).convert('L')

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)

        label = float(row['allowed'])
        return image, label
    

def generate_dataloaders(train_df, validate_df, test_df, batch_size=32, shuffle=True, transform=None):
    train_dataset = SpectrogramDataset(train_df, transform)
    validate_dataset = SpectrogramDataset(validate_df, transform)
    test_dataset = SpectrogramDataset(test_df, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validate_loader, test_loader
    
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = running_loss / total
    accuracy = correct / total

    # Flatten predictions and labels
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Compute metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    return avg_loss, accuracy, f1
    

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Voices_devkit to spectrograms"
    )
    parser.add_argument('--dataset-csv-dir',   type=Path, default='all_spectrogram_slices.csv')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=int, default=1e-4)
    parser.add_argument("--allowed-men-count", type=int, default=5)
    parser.add_argument("--allowed-women-count", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume training from a saved model.")
    parser.add_argument("--patience_lim", type=int, default=8)
    parser.add_argument("--csv-type", type=str, default='balanced')
    parser.add_argument("--CNN-model", type=str, default='CNN1')

    settings = parser.parse_args()
    epochs = settings.epochs
    batch_size = settings.batch_size
    learning_rate = settings.learning_rate
    model_name = settings.CNN_model
    csv_type = settings.csv_type
    
    if csv_type== 'all':
        df = create_all_spectrograms_classified(settings.dataset_csv_dir, settings.allowed_men_count, settings.allowed_women_count)
    elif csv_type == 'noiseless':
        df = create_noiseless_spectrograms_classified(settings.dataset_csv_dir, settings.allowed_men_count, settings.allowed_women_count)
    elif csv_type == 'balanced':
        df = create_balanced_spectrograms_classified(settings.dataset_csv_dir, settings.allowed_men_count, settings.allowed_women_count)
    elif csv_type == "balanced_noiseless":
        df = create_balanced_noiseless_spectrograms_classified(settings.dataset_csv_dir, settings.allowed_men_count, settings.allowed_women_count)

    train_df, validate_df, test_df = generate_dataframes(df)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    # creation of dataloaders
    train_loader, validate_loader, test_loader = generate_dataloaders(train_df, validate_df, test_df, batch_size=batch_size, shuffle=True, transform = transform)

    # creation of the model
    if model_name == 'CNN1':
        model = CNN1().to(device)
    elif model_name == 'CNN2':
        model = CNN2().to(device)
    else:
        print("Incorrect model type!")
        return
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if settings.resume and Path(f"{model_name}_model.pth").exists():
        print(f"🔁 Resuming training from '{model_name}_model.pth'")
        model.load_state_dict(torch.load(f"{model_name}_model.pth", map_location=device))

    best_f1 = 0.0
    patience_counter = 0
    patience_limit = settings.patience_lim  # stop if no improvement after 5 epochs
    model.to(device)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,)
        validate_loss, validate_acc, val_f1 = evaluate(model, validate_loader, criterion)

        # Early stopping check
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"{model_name}_model.pth")
            print("Model improved. Saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Early stopping patience: {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {validate_loss:.4f}, Accuracy: {validate_acc:.4f}")

    # Final test evaluation
    test_loss, test_acc , f1_val = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f} F1: {f1_val}")

    torch.save(model.state_dict(), f"{model_name}_model_final.pth")


if __name__ == "__main__":
    main()

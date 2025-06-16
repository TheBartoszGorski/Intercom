import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from myCNNs import CNN2
from train import SpectrogramDataset, create_all_spectrograms_classified, generate_dataframes, generate_dataloaders
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
from pathlib import Path
import pandas as pd


def evaluate_model_performance(model, dataloader, criterion=None, device='cpu', verbose=True):
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_loss = total_loss / total if criterion else None

    if verbose:
        print("\n===== Model Evaluation =====")
        print(f"Loss:      {avg_loss:.4f}" if avg_loss is not None else "Loss:      N/A")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model")
    parser.add_argument('--dataset-csv-dir', type=Path, default='all_speakers.csv')
    parser.add_argument('--model-path', type=Path, default='best_model.pth')
    parser.add_argument('--allowed-men-count', type=int, default=5)
    parser.add_argument('--allowed-women-count', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = create_all_spectrograms_classified(args.dataset_csv_dir, args.allowed_men_count, args.allowed_women_count)
    _, _, test_df = generate_dataframes(df)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    _, _, test_loader = generate_dataloaders(_, _, test_df, batch_size=32, shuffle=False, transform=transform)

    model = CNN2().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    criterion = nn.BCEWithLogitsLoss()

    evaluate_model_performance(model, test_loader, criterion=criterion, device=device)


if __name__ == '__main__':
    main()

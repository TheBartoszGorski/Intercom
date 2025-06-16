
from pathlib import Path

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Voices_devkit to spectrograms"
    )
    parser.add_argument('--dataset-csv-dir', type=Path, default='all_spectrograms_classified.csv')
   
    settings = parser.parse_args()
    
    df = pd.read_csv(settings.dataset_csv_dir)

    print(df[df['allowed'] == 1]['speaker'].unique())
    


if __name__ == "__main__":
    main()
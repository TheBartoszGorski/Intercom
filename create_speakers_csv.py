import pandas as pd

def main():
    data_frame_train = pd.read_csv(r"VOiCES_devkit\references\train_index.csv")
    data_frame_test = pd.read_csv(r"VOiCES_devkit\references\test_index.csv")
    # append the test dataframe to the train data frame.
    alldata_frame = pd.concat([data_frame_train, data_frame_test], axis=0, ignore_index=True)
    # Drop the mixed up index column
    alldata_frame = alldata_frame.drop(columns=["index"])

    alldata_frame.to_csv("all_speakers.csv", index=True)


if __name__ == '__main__':
    main()
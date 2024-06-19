import os
import tqdm

import pandas as pd

def get_true_labels(data_path=None):

    if data_path is None:

        print("Dataset is retrieving from...")
        print("/localscratch/neeresh/data/physionet2019/physionet.org/files/challenge-2019/1.0.0/training/training_setA/")
        
        data_path = "/localscratch/neeresh/data/physionet2019/physionet.org/files/challenge-2019/1.0.0/training/training_setA/"
    
    training_files = [file for file in os.listdir(data_path) if file.endswith('.psv')]
    training_files.sort()

    for i, file in enumerate(tqdm.tqdm(training_files, desc="Storing labels: ", total=len(training_files))):
        try:
            temp = pd.read_csv(os.path.join(data_path, file), sep='|')['SepsisLabel']
            temp.to_csv(os.path.join('./labels/', file), sep='|', index=False)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    print("Completed.")


if __name__ == '__main__':
    get_true_labels()


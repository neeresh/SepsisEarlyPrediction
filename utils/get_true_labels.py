import os
import tqdm

import pandas as pd


def get_true_labels(custom_files=None):
    if custom_files is None:

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

    else:
        print("Evaluating on custom files...")

        data_path = [
            "/localscratch/neeresh/data/physionet2019/physionet.org/files/challenge-2019/1.0.0/training/training_setA/",
            "/localscratch/neeresh/data/physionet2019/physionet.org/files/challenge-2019/1.0.0/training/training_setB/"
        ]
        all_files = []
        for file_path in data_path:
            print(f"Total number of files in {file_path}: {len(os.listdir(file_path))}")
            for patient_file in os.listdir(file_path):
                all_files.append(f"{file_path}{patient_file}")

        print("Custom Files...")
        print(f"Total number of files: {len(all_files)}")

        training_files = []
        for selected_patient in custom_files:
            for file in all_files:
                if selected_patient in file:
                    training_files.append(file)

        for i, file in enumerate(tqdm.tqdm(training_files, desc="Storing labels: ", total=len(training_files))):
            try:
                temp = pd.read_csv(os.path.join(file), sep='|')['SepsisLabel']
                file = file.split('/')[-1]
                temp.to_csv(os.path.join('./labels/', file), sep='|', index=False)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        print("Completed.")


if __name__ == '__main__':
    get_true_labels()


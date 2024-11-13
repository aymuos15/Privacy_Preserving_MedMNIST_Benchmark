import pandas as pd
import sys
sys.path.append('/home/localssk23/final_ppai/')
from config import CONFIG

datasets = [
    "breastmnist",
    "retinamnist",
    "pneumoniamnist",
    # "dermamnist", # There is a data leak here so skip
    "bloodmnist",
    # "chestmnist", # skipping cos multilabel
    "organcmnist",
    "organsmnist",
    "organamnist",
    "pathmnist",
    "octmnist",
    "tissuemnist",
]

for dataset in datasets:
    class_results = pd.read_csv(CONFIG['result_path'] + f'class_results_{dataset}.csv')
    overall_results = pd.read_csv(CONFIG['result_path'] + f'overall_results_{dataset}.csv')

    print(f'Dataset: {dataset}')
    print('Class-wise results:')
    print(class_results)
    print()
    print('Overall results:')
    print(overall_results)
    print()
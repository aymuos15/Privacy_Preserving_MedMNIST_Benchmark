from train import train, train_private
from test import test
from load import load_data
from save_results import save

from config import CONFIG

import argparse

def main(data_path):

    print(f"Processing dataset: {data_path}")

    train_loader, test_loader, class_counts, channels = load_data(data_path)

    num_classes = len(class_counts['test'])
    train_class_counts = class_counts['train']
    test_class_counts = class_counts['test']

    for i in range(CONFIG['num_runs']):

        model = train_private(train_loader, channels, num_classes)
        accuracy, auc, classwise_accuracy, classwise_auc = test(model, test_loader, num_classes)
        del model

        save(data_path, num_classes, classwise_accuracy, classwise_auc, accuracy, auc, train_class_counts, test_class_counts, i, style='private')

        model = train(train_loader, channels, num_classes)
        accuracy, auc, classwise_accuracy, classwise_auc = test(model, test_loader, num_classes)
        del model

        save(data_path, num_classes, classwise_accuracy, classwise_auc, accuracy, auc, train_class_counts, test_class_counts, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate model with specified data path and channels')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input data file')
    
    args = parser.parse_args()
    main(args.data_path)
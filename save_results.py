import pandas as pd

from config import CONFIG

def save(data_path, num_classes, classwise_accuracy, classwise_auc, accuracy, auc, train_class_counts, test_class_counts, run, style=None):
    class_results = []
    overall_results = []
    sample_info = []

    class_acc = classwise_accuracy
    class_auc = classwise_auc
    acc = accuracy

    for label in range(num_classes):
        class_results.append({
            'Dataset': data_path.split('/')[-1].split('.')[0],
            'Class': label,
            'Accuracy': class_acc.get(label, 0),
            'AUC': class_auc.get(label, 0),
        })
    
    overall_results.append({
        'Dataset': data_path.split('/')[-1].split('.')[0],
        'Overall Accuracy': acc,
        'Overall AUC': auc,
        'Balanced Accuracy': sum(class_acc.values()) / num_classes,
    })

    sample_info.append({
        'Total Train Samples': int(sum(train_class_counts)),
        'Total Test Samples': int(sum(test_class_counts)),
        **{f'Train Samples Class {i}': int(train_class_counts[i]) for i in range(num_classes)},
        **{f'Test Samples Class {i}': int(test_class_counts[i]) for i in range(num_classes)}
    })

    class_results_df = pd.DataFrame(class_results)
    overall_results_df = pd.DataFrame(overall_results)
    sample_info_df = pd.DataFrame(sample_info).T

    dataset_name = data_path.split('/')[-1].split('.')[0]

    if style == 'private':
        class_results_df.to_csv(CONFIG['result_path'] + f"class_results_{dataset_name}_run{str(run)}_private.csv", index=False)
        overall_results_df.to_csv(CONFIG['result_path'] + f"overall_results_{dataset_name}_run{str(run)}_private.csv", index=False)
        sample_info_df.to_csv(CONFIG['result_path'] + f"sample_info_{dataset_name}_run{str(run)}_private.csv")
    else:
        class_results_df.to_csv(CONFIG['result_path'] + f"class_results_{dataset_name}_run{str(run)}.csv", index=False)
        overall_results_df.to_csv(CONFIG['result_path'] + f"overall_results_{dataset_name}_run{str(run)}.csv", index=False)
        sample_info_df.to_csv(CONFIG['result_path'] + f"sample_info_{dataset_name}_run{str(run)}.csv")
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score

from config import CONFIG

device = CONFIG['device']

def test(model, test_loader, num_classes):
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            targets = targets.squeeze().long()
            _, predicted = outputs.max(1)
            probs = F.softmax(outputs, dim=1)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
    
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_outputs = np.concatenate(all_outputs)
    
    # Calculate overall accuracy
    accuracy = np.mean(all_targets == all_predictions)
    
    # Calculate classwise accuracy as a dictionary
    classwise_accuracy = {}
    for i in range(num_classes):
        class_mask = all_targets == i
        class_accuracy = np.mean(all_predictions[class_mask] == i)
        classwise_accuracy[i] = class_accuracy
    
    # Convert targets to one-hot encoding
    targets_onehot = np.eye(num_classes)[all_targets.astype(int)]
    auc = roc_auc_score(targets_onehot, all_outputs, multi_class='ovr', average='macro')
    
    # Calculate classwise AUC as a dictionary
    classwise_auc = {}
    individual_aucs = roc_auc_score(targets_onehot, all_outputs, multi_class='ovr', average=None)
    for i, class_auc in enumerate(individual_aucs):
        classwise_auc[i] = class_auc
    
    return accuracy, auc, classwise_accuracy, classwise_auc
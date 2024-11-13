import torch
import torch.nn as nn

home = '/home/soumya/final_ppai'

CONFIG = {
    'criterion': nn.CrossEntropyLoss(),

    "batch_size": 1024,
    "num_epochs": 120,

    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "result_path": f'{home}/results/',

    "num_runs": 3,

    "epsilon": 8,
    "max_grad_norm": 1.2,
}

### For nohup
#? nohup ./run.sh > output.log 2>&1 &

### to check the process
#? pgrep -f run.sh
#? kill <PID>
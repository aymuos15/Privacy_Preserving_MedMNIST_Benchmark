import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

import tqdm

from model import Net_28
from config import CONFIG

import warnings
warnings.filterwarnings("ignore")

device = CONFIG['device']
criterion = CONFIG['criterion']

def lr_lambda(epoch):
    initial_lr = 0.001  # Initial learning rate
    if epoch < 50:
        return initial_lr / initial_lr  # Learning rate remains 0.001
    elif epoch < 75:
        return 0.1 * initial_lr / initial_lr  # Delay learning rate to 0.0001 after 50 epochs
    else:
        return 0.01 * initial_lr / initial_lr  # Delay learning rate to 0.00001 after 75 epochs

def train(train_loader, channels, num_classes):

    model = Net_28(channels, num_classes)

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda)

    model.to(device)
    criterion.to(device)

    model.train()

    for epoch in tqdm.tqdm(range(CONFIG['num_epochs'])):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()

    scheduler.step()

    return model

def train_private(train_loader, channels, num_classes):
    
    MAX_GRAD_NORM = CONFIG['max_grad_norm']
    EPSILON = CONFIG['epsilon']
    DELTA = 1 / len(train_loader)

    model = Net_28(channels, num_classes)

    errors = ModuleValidator.validate(model, strict=False)
    errors[-5:]

    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    privacy_engine = PrivacyEngine(accountant='rdp')
    
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda)

    model_private, optimizer_private, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=CONFIG['num_epochs'],
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    model_private.to(device)
    criterion.to(device)

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=CONFIG['batch_size'],
        optimizer=optimizer_private
    ) as memory_safe_data_loader:

        model_private.train()

        for epoch in tqdm.tqdm(range(CONFIG['num_epochs'])):
            for inputs, targets in memory_safe_data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer_private.zero_grad()
                outputs = model(inputs)

                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                    
                loss.backward()
                optimizer_private.step()

        scheduler.step()

    return model_private
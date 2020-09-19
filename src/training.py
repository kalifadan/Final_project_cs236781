import os

import torch
from torch import nn, autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
import tqdm.notebook as tnotebook
import tqdm


def train(model, dataset, config, device=None):
    """
    Train a given model on a given dataset with configuration options.
    :param model: The model to train.
    :param dataset: The dataset
    :param config: Configuration parameters (outlined below)
    :param device: CUDA/CPU device for tensor operations
    :return: Tuple of (loss_list, acc_list)
    """

    # Configuration parameters
    num_epochs = config.get('num_epochs', 10)
    num_workers = config.get('num_workers', 0)
    batch_size = config.get('batch_size', 1)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.01)
    is_notebook = config.get('is_notebook', False)
    verbose = config.get('verbose', False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Preparation
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    # Train the model
    loss_list = []
    acc_list = []
    trange = tnotebook.trange if is_notebook else tqdm.trange
    tq = tnotebook.tqdm if is_notebook else tqdm.tqdm

    r = trange(num_epochs, desc='Epoch')
    for epoch in r:
        acc = 0
        total_size = len(dataset)
        iteration = 0

        r_inner = tq(dataloader, desc='Iteration')
        for batch_data, batch_labels in r_inner:
            optimizer.zero_grad()
            X, y_true = batch_data.to(device), batch_labels.to(device)

            # Forward-propagate
            output = model(X)

            # if any(torch.isnan(output)):
            #     for name, param in model.named_parameters():
            #         if param.requires_grad:
            #             print(name, param.data)

            # Calculate loss
            loss = criterion(output, y_true)

            # Back-propagate and perform optimisation
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            _, prediction = torch.max(output.data, 1)

            correct = (prediction == y_true).sum().item()
            if verbose:
                print('Ground truth:', batch_labels[:100])
                print('Prediction:', prediction[:100])
            acc += correct
            iteration += 1
            r_inner.set_description('Current loss: {:.2f}'.format(loss_list[-1]))

        acc = acc / total_size
        acc_list.append(acc)
        r.set_description('Accuracy: {:.2f}%'.format(acc * 100))
        print('Epoch [{}/{}], Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, acc * 100))

    return loss_list, acc_list


def test(model, dataset, config, device=None):
    """
    Test a pretrained model on a given test dataset
    :param model: The model to train.
    :param dataset: The dataset
    :param config: Configuration parameters (outlined below)
    """
    print('Testing model...')
    # Configuration parameters
    num_workers = config.get('num_workers', 0)
    is_notebook = config.get('is_notebook', False)
    verbose = config.get('verbose', False)

    # Preparation
    dataloader = DataLoader(dataset, num_workers=num_workers)

    # Train the model
    acc = 0
    tq = tnotebook.tqdm if is_notebook else tqdm.tqdm
    total_size = len(dataset)
    iterator = iter(dataloader)
    y_pred = []
    with torch.no_grad():
        for batch_data, batch_labels in tq(iterator, desc='Example'):
            X, y_true = batch_data.to(device), batch_labels.to(device)
            output = model(X)

            # Track the accuracy
            prediction = output.argmax(dim=1).item()
            y_pred += [prediction]
            correct = (prediction == y_true).sum().item()
            acc += correct

            if verbose:
                print('Ground truth:', batch_labels[0])
                print('Prediction:', prediction)

    acc = acc / total_size
    print('Accuracy: {:.2f}%'.format(acc * 100))
    return y_pred, acc


def save_model(model, config, checkpoint_file, checkpoint_dir='./checkpoints'):
    path = os.path.join(checkpoint_dir, checkpoint_file)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save({
        'state_dict': model.state_dict(),
        'config': config
    }, path)


def load_model(model: nn.Module, checkpoint_file, checkpoint_dir='./checkpoints'):
    path = os.path.join(checkpoint_dir, checkpoint_file)
    loaded = torch.load(path)
    state_dict = loaded['state_dict']
    model.load_state_dict(state_dict)
    return model, loaded['config']
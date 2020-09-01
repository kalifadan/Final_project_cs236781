import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
import tqdm.notebook as tnotebook
import tqdm


def train(model, dataset, config):
    # total_size = len(x_train)
    # test.assertEqual(total_size, len(y_train))

    # Loss and optimizer
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    is_notebook = config['is_notebook']
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Preparation
    # sampler = torch.utils.data.sampler.WeightedRandomSampler([0.7, 0.3], len(dataset), replacement=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Train the model
    loss_list = []
    acc_list = []
    trange = tnotebook.trange if is_notebook else tqdm.trange
    tq = tnotebook.tqdm if is_notebook else tqdm.tqdm
    for epoch in trange(num_epochs, desc='Epoch'):
        acc = 0
        total_size = len(dataset)
        for batch_data, batch_labels in tq(dataloader, desc='Iteration'):
            optimizer.zero_grad()
            batch_data.requires_grad = True
            # for b in batch_data:
            #     print(b.sum().item(), end=', ')
            print(batch_data[:, 0, :].shape)
            output = model(batch_data[:, 0, :].unsqueeze(1).float())

            # print(output[:10])
            # print('Labels:', batch_labels)

            loss = criterion(output, batch_labels)
            print('Loss: ', loss)

            # Backprop and perform optimisation
            loss_list.append(loss.item())
            loss.backward()
            # print('dx/dy =', autograd.grad(loss, batch_data))
            optimizer.step()

            # Track the accuracy
            #         probability = torch.distributions.categorical.Categorical(output)
            #         prediction = probability.sample()
            prediction = output.softmax(dim=1).argmax(dim=1)

            print('Output:', output[:50])
            print('Ground truth:', batch_labels[:50])
            print('Prediction:', prediction[:50])

            correct = (prediction == batch_labels).sum().item()
            print('Correct: {}'.format(correct))
            acc += correct

        acc = acc / (total_size)
        acc_list.append(acc)
        print('Epoch [{}/{}], Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, acc * 100))


def test(model, dataset, config):
    print('Testing model...')
    # Loss and optimizer
    num_workers = config['num_workers']
    is_notebook = config['is_notebook']

    # Preparation
    dataloader = DataLoader(dataset, num_workers=num_workers)

    # Train the model
    acc = 0
    tq = tnotebook.tqdm if is_notebook else tqdm.tqdm
    total_size = len(dataset)
    iterator = iter(dataloader)
    y_pred = []
    for batch_data, batch_labels in tq(iterator, desc='Example'):
        output = model(batch_data)

        # Track the accuracy
        prediction = output.argmax(dim=1).item()
        y_pred += [prediction]
        correct = (prediction == batch_labels).sum().item()
        # print('Correct: {}'.format(correct))
        acc += correct

    acc = acc / total_size
    print('Accuracy: {:.2f}%'.format(acc * 100))
    return y_pred, acc

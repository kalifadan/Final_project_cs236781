import torch
from torch import nn, autograd
from torch.autograd import Variable
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
    class_weights = config['class_weights']
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Preparation
    if class_weights is not None:
        print(len(class_weights))
        print(len(dataset))
        assert len(class_weights) == len(dataset)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights, len(dataset), replacement=True)
    else:
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
            data = Variable(batch_data)
            labels = Variable(batch_labels)
            # batch_data.requires_grad = True
            # for b in batch_data:
            #     print(b.sum().item(), end=', ')
            # print('Batch: ', data.shape)
            output = model(data)

            # print(output[:10])
            # print('Labels:', batch_labels)

            loss = criterion(output, labels)

            # if iteration % 8 == 0:
            #     print('Loss: ', loss)

            # Backprop and perform optimisation
            loss_list.append(loss.item())
            loss.backward()
            # print('dx/dy =', autograd.grad(loss, batch_data))
            optimizer.step()

            # Track the accuracy
            #         probability = torch.distributions.categorical.Categorical(output)
            #         prediction = probability.sample()
            _, prediction = torch.max(output.data, 1)

            correct = (prediction == batch_labels).sum().item()
            # if iteration % 8 == 0:
            #     print('Ground truth:', batch_labels[:50])
            #     print('Prediction:', prediction[:50])
            #     print('Correct: {}'.format(correct))
            acc += correct
            iteration += 1
            r_inner.set_description('Current loss: {:.2f}'.format(loss_list[-1]))

        acc = acc / total_size
        acc_list.append(acc)
        r.set_description('Accuracy: {:.2f}%'.format(acc * 100))
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

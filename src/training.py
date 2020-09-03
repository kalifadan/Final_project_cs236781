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
    num_epochs = config.get('num_epochs', 10)
    num_workers = config.get('num_workers', 0)
    batch_size = config.get('batch_size', 1)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.01)
    is_notebook = config.get('is_notebook', False)
    # class_weights = config.get('class_weights', None)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
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
            data = Variable(batch_data)
            labels = Variable(batch_labels)
            # batch_data.requires_grad = True
            # for b in batch_data:
            #     print(b.sum().item(), end=', ')
            # print('Batch: ', data.shape)
            output = model(data)

            print('Output:', output[:10])
            # print('Labels:', batch_labels)
            # print(labels.shape)

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
            print('Ground truth:', batch_labels[:10])
            print('Prediction:', prediction[:10])
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
        # print('Output: ', output[batch_labels == 1], batch_labels[batch_labels == 1])

        # Track the accuracy
        prediction = output.argmax(dim=1).item()
        y_pred += [prediction]
        correct = (prediction == batch_labels).sum().item()
        # print('Correct: {}'.format(correct))
        acc += correct

    acc = acc / total_size
    print('Accuracy: {:.2f}%'.format(acc * 100))
    return y_pred, acc

import torch
from torch import nn
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
    criterion = nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Preparation
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
            output = model(batch_data)

            print(output[:10])
            print('Labels:', batch_labels)

            loss = criterion(output[:, 1], batch_labels.float())

            # Backprop and perform optimisation
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            # Track the accuracy
            #         probability = torch.distributions.categorical.Categorical(output)
            #         prediction = probability.sample()
            prediction = output.argmax(dim=1)

            print('Output:', output)
            print('Ground truth:', batch_labels)
            print('Prediction:', prediction)
            correct = (prediction == batch_labels).sum().item()
            print('Correct: {}'.format(correct))
            acc += correct

        acc = acc / (total_size)
        acc_list.append(acc)
        print('Epoch [{}/{}], Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, acc * 100))
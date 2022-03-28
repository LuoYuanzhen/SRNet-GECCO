import json
import random

import torch
from torch.autograd import Variable

from data_utils import io, draw
from dataset_config import FUNC_MAP, INTER_MAP
from neural_networks.nn_models import MLP2, MLP3


def data_iter(batch_size, x, y):
    num_samples = x.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_samples)])
        yield x.index_select(0, j), y.index_select(0, j)


def split_dataset(dataset):
    prob = 0.8
    n_train = int(dataset.shape[0] * prob)

    return dataset[:n_train], dataset[n_train:]


def evaluate_loss(data_iters, net, loss_func):
    metric_sum, n = 0.0, 0
    dev = list(net.parameters())[0].device
    with torch.no_grad():
        for x, y in data_iters:
            pred = net(x.to(dev))
            if isinstance(pred, tuple):
                pred = pred[-1]

            metric_sum += loss_func(pred, y.to(device)).cpu().item()
            n += 1

    if n == 0:
        return 0
    return metric_sum / n


def predict(net, x):
    dev = list(net.parameters())[0].device
    return list([output.cpu().detach() for output in net(x.to(dev))])


if __name__ == '__main__':

    # ediable hyparameters
    data_filename = 'feynman5'
    n_output, n_hidden, lr, n_epoch, batch_size = 1, 5, 1e-3, 50000, 300
    optimizer = torch.optim.Adam
    reg_model = MLP3

    # processing data
    dataset = io.get_dataset(f'../dataset/{data_filename}')
    data_train, data_test = split_dataset(dataset)
    n_var = data_train.shape[1] - n_output
    x_train, y_train = data_train[:, :n_var], data_train[:, n_var:]
    x_test, y_test = data_test[:, :n_var], data_test[:, n_var:]

    # train net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse = torch.nn.MSELoss()

    reg_model = reg_model(n_var, n_output, n_hidden).to(device)
    optimizer = optimizer(reg_model.parameters(), lr=lr)

    train_loss, test_loss = [], []
    print('start')
    for epoch in range(n_epoch):
        train_loss_sum, num_batchs = 0.0, 0
        train_iter = data_iter(batch_size, x_train, y_train)
        for x_batch, y_batch in train_iter:
            if torch.cuda.is_available():
                x_batch = Variable(x_batch).to(device)
                y_batch = Variable(y_batch).to(device)

            y_hat = reg_model(x_batch)
            if isinstance(y_hat, tuple):
                y_hat = y_hat[-1]

            loss = mse(y_hat, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            num_batchs += 1
        train_aver_loss = train_loss_sum / num_batchs
        test_aver_loss = evaluate_loss(data_iter(batch_size, x_test, y_test), reg_model, mse)

        train_loss.append(train_aver_loss)
        test_loss.append(test_aver_loss)
        print('# epoch %d: Train Loss=%.8f, Test Loss=%.8f' %
              (epoch + 1, train_aver_loss, test_aver_loss))

    # draw and save
    result_dir = f'../dataset/{data_filename}_nn/'
    io.mkdir(result_dir)

    draw.draw_mul_curves_and_save(list(range(n_epoch)), [train_loss, test_loss],
                                  savepath=result_dir+'trend.pdf',
                                  title='Train and Test loss trend',
                                  xlabel='epoch',
                                  ylabel='loss',
                                  labels=['Train', 'Test'])

    x, y = dataset[:, :n_var], dataset[:, -1]
    layers = [x] + predict(reg_model, x)
    io.save_layers(layers, save_dir=result_dir)
    io.save_nn_model(reg_model, savepath=f'{result_dir}nn_module.pt', save_type='dict')

    draw.project_to_2d_and_save(vars=tuple([dataset[:, i] for i in range(n_var)]), zs=(y, layers[-1]),
                                savefile=f'{result_dir}compare.pdf',
                                zs_legends=['true', 'nn'])

    # draw extrapolate and interpolate data
    x_ranges = INTER_MAP[data_filename]
    original_func = FUNC_MAP[data_filename]

    draw.draw_polate_data_curves(x_ranges=x_ranges,
                                 models=[reg_model.cpu()],
                                 original_func=original_func,
                                 model_labels=['nn'],
                                 savepath=f'{result_dir}polate.pdf',
                                 title='interpolate and extrapolate')

    # Save hyperparameters
    log_dict = {
        'neurons': list([layer.shape[1] for layer in layers]),
        'n_epoch': n_epoch,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'lr': lr,
        'mlp_model': reg_model.__class__.__name__
    }
    with open(f'{result_dir}settings.json', 'w') as f:
        json.dump(log_dict, f, indent=4)







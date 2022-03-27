from functools import partial

from torch import nn


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hiddens, activation_func=nn.Sigmoid(), regression=True):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens

        if isinstance(n_hiddens, list) or isinstance(n_hiddens, tuple):
            in_features = self.n_input
            for num, n_hidden in enumerate(n_hiddens):
                setattr(self, 'fc{}'.format(num+1),
                        nn.Sequential(
                            nn.Linear(in_features, n_hidden),
                            activation_func
                        ))
                in_features = n_hidden
            if regression:
                setattr(self, 'fc{}'.format(len(n_hiddens)+1),
                    nn.Linear(in_features, self.n_output)
                )
            else:
                setattr(self, 'fc{}'.format(len(n_hiddens)+1),
                        nn.Sequential(
                            nn.Linear(in_features, self.n_output),
                            nn.Softmax(dim=1)
                        ))
        else:
            raise ValueError("expected n_hiddens as list or tuple, not {}".format(type(n_hiddens)))

    def forward(self, x):
        inputs = self.fc1(x.reshape(x.shape[0], -1))
        outputs = [inputs]

        n_layer = len(self.n_hiddens)
        for i in range(2, n_layer+2):
            inputs = getattr(self, 'fc{}'.format(i))(inputs)
            outputs.append(inputs)

        return tuple(outputs)


class LeNet(nn.Module):
    def __init__(self):
        # with input image shape: Bx1x28x28
        super(LeNet, self).__init__()
        self.n_channels = [1, 6, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv_input_size_list = [(28, 28), (12, 12)]  # shapes of input for each Conv2d
        self.conv_output_size_list = [(24, 24), (8, 8)]  # shapes of output for each conv layer
        self.n_input = 16 * 4 * 4
        self.n_hiddens = [120, 84]
        self.n_output = 10
        in_features = self.n_input
        for num, n_neuron in enumerate(self.n_hiddens):
            setattr(self, 'fc{}'.format(num + 1),
                    nn.Sequential(
                        nn.Linear(in_features, n_neuron),
                        nn.ReLU()
                    ))
            in_features = n_neuron
        setattr(self, 'fc{}'.format(len(self.n_hiddens)+1), nn.Sequential(
            nn.Linear(in_features, self.n_output),
            nn.Softmax(dim=1)
        ))

    def forward(self, X):
        conved1 = self.conv1(X)
        conved2 = self.conv2(conved1)
        outputs = [conved1, conved2]

        inputs = conved2.reshape(conved2.shape[0], -1)
        n_layer = len(self.n_hiddens) + 1
        for i in range(1, n_layer + 1):
            inputs = getattr(self, 'fc{}'.format(i))(inputs)
            outputs.append(inputs)

        return outputs


NN_MAP = {
    'kkk0': MLP(1, 1, [3, 3]),
    'kkk1': MLP(2, 1, [3, 3]),
    'kkk2': MLP(1, 1, [5, 5]),
    'kkk3': MLP(2, 1, [4, 4, 4]),
    'kkk4': MLP(3, 1, [4, 4]),
    'kkk5': MLP(2, 1, [5, 5]),
    'feynman0': MLP(3, 1, [3, 3]),
    'feynman1': MLP(4, 1, [3, 3]),
    'feynman2': MLP(5, 1, [3, 3]),
    'feynman3': MLP(2, 1, [3, 3]),
    'feynman4': MLP(5, 1, [5, 5]),
    'feynman5': MLP(5, 1, [5, 5, 5]),

    'adult': partial(MLP, regression=False),
    'agaricus_lepiota': partial(MLP, regression=False),
    'analcatdata_aids': partial(MLP, regression=False),
    'breast': partial(MLP, regression=False),
    'car': partial(MLP, regression=False),
    'binary_ball': partial(MLP, regression=False),
    'moons': partial(MLP, regression=False),
    'three_class': partial(MLP, regression=False),
    'four_class': partial(MLP, regression=False),
    'digit': partial(MLP, regression=False)
}

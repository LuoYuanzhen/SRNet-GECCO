from torch import nn


class MLP2(nn.Module):
    def __init__(self, num_input, num_output, n_hidden):
        super(MLP2, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.n_hidden = n_hidden

        self.fc1 = nn.Sequential(
            nn.Linear(num_input, n_hidden),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
        )
        self.fc3 = nn.Linear(n_hidden, num_output)

    def forward(self, x):
        output1 = self.fc1(x)
        output2 = self.fc2(output1)
        outputs = self.fc3(output2)
        return output1, output2, outputs


class MLP3(nn.Module):
    def __init__(self, num_input, num_output, n_hidden):
        super(MLP3, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.n_hidden = n_hidden

        self.fc1 = nn.Sequential(
            nn.Linear(num_input, n_hidden),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
        )
        self.fc4 = nn.Linear(n_hidden, num_output)

    def forward(self, x):
        output1 = self.fc1(x)
        output2 = self.fc2(output1)
        output3 = self.fc3(output2)
        outputs = self.fc4(output3)
        return output1, output2, output3, outputs


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
        inputs = self.fc1(x)
        outputs = [inputs]

        n_layer = len(self.n_hiddens)
        for i in range(2, n_layer+2):
            inputs = getattr(self, 'fc{}'.format(i))(inputs)
            outputs.append(inputs)

        return tuple(outputs)


NN_MAP = {
    'kkk0': MLP2(1, 1, 3),
    'kkk1': MLP2(2, 1, 3),
    'kkk2': MLP2(1, 1, 5),
    'kkk3': MLP3(2, 1, 4),
    'kkk4': MLP2(3, 1, 4),
    'kkk5': MLP2(2, 1, 5),
    'feynman0': MLP2(3, 1, 3),
    'feynman1': MLP2(4, 1, 3),
    'feynman2': MLP2(5, 1, 3),
    'feynman3': MLP2(2, 1, 3),
    'feynman4': MLP2(5, 1, 5),
    'feynman5': MLP3(5, 1, 5),
    'adult': MLP(14, 2, [600, 600, 600]),
    'agaricus_lepiota': MLP3(22, 2, 600)
}

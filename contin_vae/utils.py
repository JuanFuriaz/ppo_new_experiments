import math
import torch


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndim):
        super(Unflatten, self).__init__()
        self.ndim = ndim

    def forward(self, x):
        return x.view(x.size(0), self.ndim, 1, 1)


class LinearClassifier(torch.nn.Module):
    def __init__(self, num_hidden, num_classes):
        super(LinearClassifier, self).__init__()
        self.map = torch.nn.Linear(num_hidden, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


class NonlinearClassifier(torch.nn.Module):
    def __init__(self, num_hidden, num_classes):
        super(NonlinearClassifier, self).__init__()
        self.map = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
            torch.nn.BatchNorm1d(num_features=num_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=num_hidden, out_features=num_classes, bias=True)
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


class LinearRegressor(torch.nn.Module):
    def __init__(self, num_hidden):
        super(LinearRegressor, self).__init__()
        self.map = torch.nn.Linear(num_hidden, 1)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x).view(-1)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


class NonlinearRegressor(torch.nn.Module):
    def __init__(self, num_hidden):
        super(NonlinearRegressor, self).__init__()
        self.map = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
            torch.nn.BatchNorm1d(num_features=num_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=num_hidden, out_features=1, bias=True)
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x).view(-1)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


def init_weights(self):
    # see also https://github.com/pytorch/pytorch/issues/18182
    for m in self.modules():
        if type(m) in {
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
            torch.nn.Linear,
        }:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                torch.nn.init.normal_(m.bias, -bound, bound)

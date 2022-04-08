import crypten.nn as nn


class Crypten_Net_mnist(nn.Module):
    def __init__(self):
        super(Crypten_Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.max_pool2d(self.conv1(x), 2).relu()
        x = self.max_pool2d(self.conv2_drop(self.conv2(x)), 2).relu()
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x).relu()
        x = self.dropout(x, training=self.training)
        x = self.fc2(x)
        return x.log_softmax(dim=1)


class Crypten_Net_cifar(nn.Module):
    def __init__(self):
        super(Crypten_Net_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x.log_softmax(dim=1)

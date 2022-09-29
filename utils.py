from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torch import autograd
from torch.autograd import Variable


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


# Define CNN model
class CNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_target):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # (1, 28, 28) -> (32, 26, 26)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  # (32, 26, 26) -> (32, 24, 24)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 24, 24) -> (32, 12, 12)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # (32, 12, 12) -> (64, 10, 10)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)  # (64, 10, 10) -> (64, 8, 8)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 8, 8) -> (64, 4, 4)

        self.fc1 = nn.Linear(1024, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_target)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(-1, 1024)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=400, hidden_dropout=True, input_dropout=True):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dropout_prob = 0.5 if hidden_dropout else 0
        self.input_dropout_prob = 0.2 if input_dropout else 0

        # input layer
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.input_dropout_prob)
        # hidden layers
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.hidden_dropout_prob)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.hidden_dropout_prob)
        # output layer
        self.fc4 = nn.Linear(hidden_size, self.output_size)

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        output = self.fc4(out)
        return output.reshape(-1, self.output_size)


class EWC_H(object):
    def __init__(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, batch_size=32, sample_size=1000):

        self.model = model
        self.data_loader = data_loader
        self.params = {n: p for n, p in self.model.named_parameters()}

        # Consolidation
        self._means = {}
        for n, p in self.params.items():
            self._means[n] = p.data.clone()
        self._fisher_matrices = self._diag_fisher(batch_size, sample_size)

    def _diag_fisher(self, batch_size, sample_size):
        # Device will determine whether to run the training on GPU or CPU.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        fisher_matrices = {}
        for n, p in self.params.items():
            fisher_matrices[n] = p.data.clone()
            fisher_matrices[n].data.zero_()

        log_likelihoods = []
        for data, labels in self.data_loader:
            data = data.to(device)
            data = data.view(batch_size, -1)
            output = self.model(data)
            log_likelihood = F.log_softmax(output, dim=1)[range(batch_size), labels.data]
            log_likelihoods.append(log_likelihood)
            if len(log_likelihoods) >= sample_size // batch_size:
                break
        log_likelihoods = torch.cat(log_likelihoods).unbind()

        log_likelihoods_grads = zip(*[autograd.grad(
            like, self.params.values(), retain_graph=(i < len(log_likelihoods))
        ) for i, like in enumerate(log_likelihoods, 1)])
        log_likelihoods_grads = [torch.stack(gs) for gs in log_likelihoods_grads]

        fisher_diagonals = [(g ** 2).mean(0) for g in log_likelihoods_grads]
        param_names = list(self.params.keys())
        fisher_matrices = {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}
        return fisher_matrices

    def penalty(self, model: nn.Module, lamda=40):
        losses = []
        for n, p in model.named_parameters():
            losses.append((self._fisher_matrices[n] * (p - self._means[n]) ** 2).sum())
        return (lamda / 2) * sum(losses)

    def debugPrint(self):
        for n, p in self._fisher_matrices.items():
            print(n)
            print(p)
            print("=============")


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items(): # # dict.items() = key, value iteration
            self._means[n] = variable(p.data)
            # # type(p) = torch.nn.parameter.Parameter, type(p.data) = torch.Tensor
            # # 실제 p and p.data의 contents는 동일함 (tensor 형태 행렬)

    def _diag_fisher(self):
        # Device will determine whether to run the training on GPU or CPU.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        # # So far, precision_matrices는 {param_name :
        # # tensor of equal size to each parameter but all elements are 0}

        self.model.eval()
        for image in self.dataset: # single batch dataset
            self.model.zero_grad()
            image = variable(image)
            image = image.to(device)

            output = self.model(image)
            output = output.view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()  # type(_loss) = torch.tensor
        return loss

    def debugPrint(self):
        for n, p in self._precision_matrices.items():
            print(n)
            print(p)
            print("===")


def training(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
             accuracy_list: list, epoch, task_idx):
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    train_ac = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)
        loss = F.cross_entropy(output, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Softmax Layer
        y_prob = nn.Softmax(dim=1)(output)
        y_pred = torch.argmax(y_prob, dim=1)

        # Accuracy
        train_ac += (y_pred.eq(labels).float()).sum().item()

    avg_train_ac = train_ac / len(data_loader.dataset)
    accuracy_list.append(avg_train_ac)

    if (epoch + 1) % 5 == 0:
        print('Epoch_{}: task_{} Train_Acc: {train_ac:.4f}'
              .format(epoch + 1, task_idx, train_ac=avg_train_ac))

    return accuracy_list


def training_ewc(model: nn.Module, ewc: EWC_H, lamda: int, optimizer: torch.optim,
                 data_loader: torch.utils.data.DataLoader, accuracy_list: list, epoch, task_idx):
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    train_ac = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)
        ce_loss = F.cross_entropy(output, labels)
        ewc_loss = ewc.penalty(model, lamda=lamda)
        loss = ce_loss + ewc_loss

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Softmax Layer
        y_prob = nn.Softmax(dim=1)(output)
        y_pred = torch.argmax(y_prob, dim=1)

        # Accuracy
        train_ac += (y_pred.eq(labels).float()).sum().item()

    avg_train_ac = train_ac / len(data_loader.dataset)
    accuracy_list.append(avg_train_ac)

    if (epoch + 1) % 5 == 0:
        print('Epoch_{}: task_{} Train_Acc: {train_ac:.4f}'
              .format(epoch + 1, task_idx, train_ac=avg_train_ac))

    return accuracy_list


def testing(model: nn.Module, data_loader: torch.utils.data.DataLoader,
            accuracy_list: list, epoch, task_idx):
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    test_ac = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            _, predicted = torch.max(output.data, 1)

            # Softmax Layer
            y_prob = nn.Softmax(dim=1)(output)

            # Accuracy
            y_pred = torch.argmax(y_prob, dim=1)
            test_ac += (y_pred.eq(labels).float()).sum().item()

    avg_test_ac = test_ac / len(data_loader.dataset)
    accuracy_list.append(avg_test_ac)

    if (epoch + 1) % 5 == 0 and task_idx != 0:
        print('Epoch_{}: task_{} Test_Acc: {test_ac:.4f}'
              .format(epoch + 1, task_idx, test_ac=avg_test_ac))
    elif task_idx == 0:
        print('Epoch_{}: Overall Test_Acc: {test_ac:.4f}'
              .format(epoch + 1, task_idx, test_ac=avg_test_ac))

    return accuracy_list


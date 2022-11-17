import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch import autograd
from copy import deepcopy
from tabgan.sampler import GANGenerator

def h5toDF(dest_path, filename):
    temp_h5 = pd.HDFStore(dest_path + filename, 'r')
    df_temp = temp_h5.get('/my_key')
    return df_temp

class ids2018Dataset(Dataset):
    def __init__(self, df_X, enc_y):
        self.inp = torch.tensor(df_X.values)
        self.outp = torch.tensor(enc_y)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        X = self.inp[idx]
        y = self.outp[idx]
        return X, y

def buildCombinedDataset(dest_path, task_num, encoder, df_X_gen=None, df_y_gen=None):
    if (task_num == 1):
        df_X1_train = h5toDF(dest_path, 'X_train_1.h5')
        df_y1_train = h5toDF(dest_path, 'y_train_1.h5')
        df_y1_train_encoded = pd.DataFrame(encoder.transform(df_y1_train))
        df_X1_val = h5toDF(dest_path, 'X_val_1.h5')
        return df_X1_train, df_y1_train_encoded, df_X1_val
    elif (df_X_gen is not None and df_y_gen is not None):
        # Gather task-specific data
        df_X_train = h5toDF(dest_path, 'X_train_{}.h5'.format(task_num))
        df_y_train = h5toDF(dest_path, 'y_train_{}.h5'.format(task_num))
        df_y_train_encoded = pd.DataFrame(encoder.transform(df_y_train))
        df_X_val = h5toDF(dest_path, 'X_val_{}.h5'.format(task_num))

        # Combine task-specific real training data with generated data
        # Take samples of generated data and match X to y
        sample_size = min(len(df_X_train), len(df_X_gen)) // 2
        sample_X_gen = df_X_gen.sample(n=sample_size, random_state=42)
        idx_gen = sample_X_gen.index.to_list()
        sample_y_gen = df_y_gen.iloc[idx_gen]

        # Take samples of real data and match X to y
        sample_X_real = df_X_train.sample(n=sample_size, random_state=42)
        idx_real = sample_X_real.index.to_list()
        sample_y_real = df_y_train_encoded.iloc[idx_real]

        # Combine
        combined_X_train = pd.concat([sample_X_gen, sample_X_real])
        combined_y_train = pd.concat([sample_y_gen, sample_y_real])

        # Combine task-specific real validation data with generated data
        # This will only be used to help train the generator
        sample_size = min(len(df_X_val), len(df_X_gen)) // 2
        sample_X_gen = df_X_gen.sample(n=sample_size, random_state=42)
        sample_X_real = df_X_val.sample(n=sample_size, random_state=42)
        combined_X_val = pd.concat([sample_X_gen, sample_X_real])
        return combined_X_train, combined_y_train, combined_X_val
    else:
        raise Exception("buildCombinedDataset: If task_num != 1, generated data must be provided.")

def buildDataset(dsType, dest_path, num_task, encoder):
    tk_datasets = []
    if num_task == 1:
        df_X_temp = h5toDF(dest_path, 'X_{}.h5'.format(dsType))
        df_y_temp = h5toDF(dest_path, 'y_{}.h5'.format(dsType))
        enc_y_temp = encoder.transform(df_y_temp)
        t_dataset = ids2018Dataset(df_X_temp, enc_y_temp)
        return t_dataset
    else:
        for idx in range(num_task):
            df_X_temp = h5toDF(dest_path, 'X_{}_{}.h5'.format(dsType, idx + 1))
            df_y_temp = h5toDF(dest_path, 'y_{}_{}.h5'.format(dsType, idx + 1))
            enc_y_temp = encoder.transform(df_y_temp)
            t_idx_dataset = ids2018Dataset(df_X_temp, enc_y_temp)
            tk_datasets.append(t_idx_dataset)
        return tk_datasets


def data_size_check(data_loaders: list, dataset_type: str):
    num_tasks = len(data_loaders)
    print('Data size check of {} DataLoaders'.format(dataset_type))
    for i in range(num_tasks):
        print('Task_{}:'.format(i + 1))

        for data, labels in data_loaders[i]:
            print('t{}_{}:'.format(i + 1, dataset_type))
            print('# of samples: {}'.format(len(data_loaders[i].dataset)))
            print('data size: {}'.format(data.size()))  # torch.Size([32, 1, 28, 28])
            print('data[0] size: {}'.format(data[0].size()))  # torch.Size([1, 28, 28])
            print('labels size: {}'.format(labels.size()))  # torch.Size([32])
            print()
            break

    print("------\\--------")


##-------------------------------------------------------------------


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


##-------------------------------------------------------------------


def training(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
             accuracy_list: list, epoch, task_idx):
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    train_ac = 0

    for Xs, labels in data_loader:
        Xs = Xs.to(device)
        labels = labels.long()
        labels = labels.to(device)

        # Forward pass
        output = model(Xs)
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


def testing(model: nn.Module, data_loader: torch.utils.data.DataLoader,
            accuracy_list: list, epoch, task_idx):
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    test_ac = 0

    with torch.no_grad():
        for Xs, labels in data_loader:
            Xs = Xs.to(device)
            labels = labels.to(device)

            output = model(Xs)
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

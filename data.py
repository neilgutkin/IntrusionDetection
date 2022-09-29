import random
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    b, c, d = image.size()
    image = image.view(b, -1)
    image = image[:, permutation]
    image.view(b, c, d)
    return image


def get_permuted_dataset(name, train=True, download=True, permutation=None):
    dataset_class = AVAILABLE_DATASETS[name]
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    return dataset_class(
        './datasets/{name}'.format(name=name), train=train,
        download=download, transform=dataset_transform,
    )


AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST
}

AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        # transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
}

DATASET_CONFIGS = {
    'mnist': {'size': 28, 'channels': 1, 'classes': 10}
}

def get_permuted_data_loader(dataset, batch_size, cuda=False, collate_fn=None):

    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=(collate_fn or default_collate),
        **({'num_workers': 2, 'pin_memory': True} if cuda else {})
    )


class SplitMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.t_images = []
        self.t_labels = []

        for images, labels in dataset:
            images = images.reshape(-1, 28*28)
            self.t_images.append(images)
            self.t_labels.append(labels)

        self.label_0 = []
        self.label_1 = []
        self.label_2 = []
        self.label_3 = []
        self.label_4 = []
        self.label_5 = []
        self.label_6 = []
        self.label_7 = []
        self.label_8 = []
        self.label_9 = []
        self.total_set = [self.label_0, self.label_1, self.label_2, self.label_3, self.label_4, self.label_5,
                          self.label_6, self.label_7, self.label_8, self.label_9]

    def __len__(self):
        return len(self.t_labels)

    def __getitem__(self, idx):
        image = self.t_images[idx]
        label = self.t_labels[idx]
        return (image, label)

    def sort(self):
        # sort by class -> make 10 sub-datasets
        for idx, label in enumerate(self.t_labels):
            if label == 0:
                self.label_0.append((self.t_images[idx], 0))
            elif label == 1:
                self.label_1.append((self.t_images[idx], 1))
            elif label == 2:
                self.label_2.append((self.t_images[idx], 2))
            elif label == 3:
                self.label_3.append((self.t_images[idx], 3))
            elif label == 4:
                self.label_4.append((self.t_images[idx], 4))
            elif label == 5:
                self.label_5.append((self.t_images[idx], 5))
            elif label == 6:
                self.label_6.append((self.t_images[idx], 6))
            elif label == 7:
                self.label_7.append((self.t_images[idx], 7))
            elif label == 8:
                self.label_8.append((self.t_images[idx], 8))
            elif label == 9:
                self.label_9.append((self.t_images[idx], 9))

    def class_incremental_split(self, num_tasks, num_classes):
        # integrate (10/num_tasks number) of sub-datasets
        tasks = []
        num_classes_perTask = int(num_classes / num_tasks)

        for i in range(num_tasks):
            task = []
            tasks.append(task)

        iter = 0
        for task in tasks:
            iter += 1
            for i in range(num_classes_perTask):
                task.extend(self.total_set[0])
                self.total_set.pop(0)

        return tasks

    def domain_incremental_split(self):
        pass
        # random split

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [image for image in self.t_images[sample_idx]]


def data_size_check(data_loaders: list, dataset_type: str):
    num_tasks = len(data_loaders)
    print('Data size check of {} DataLoaders'.format(dataset_type))
    for i in range(num_tasks):
        print('Task_{}:'.format(i + 1))

        for data, labels in data_loaders[i]:
            print('t{}_{}:'.format(i + 1, dataset_type))
            print('# of samples: {}'.format(len(data_loaders[i].dataset)))
            print('data size: {}'.format(data.size()))    # torch.Size([32, 1, 28, 28])
            print('data[0] size: {}'.format(data[0].size()))   # torch.Size([1, 28, 28])
            print('labels size: {}'.format(labels.size()))    # torch.Size([32])
            print()
            break

    print("------\\--------")

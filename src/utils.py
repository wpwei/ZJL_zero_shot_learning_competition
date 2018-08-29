import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy


class ZJLSet(Dataset):
    def __init__(self, image_root, images, transform, labels=None, attributes=None, word_embeddings=None, label_code=None):
        self.image_root = image_root
        self.images = images
        self.labels = labels
        self.attributes = attributes
        self.word_embeddings = word_embeddings
        self.label_code = label_code
        self.transform = transform


        # transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomResizedCrop(64, (0.3, 1)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        sample = {
            'image': self.transform(Image.open(f'{self.image_root}/{self.images[item]}').convert('RGB'))
        }
        if self.labels is not None:
            sample['label'] = self.labels[item]
            sample['attributes'] = self.attributes[sample['label']]
            sample['word_embeddings'] = self.word_embeddings[sample['label']]


        return sample


def normalize(data, method='l2'):
    if method == 'l2':
        return data / np.linalg.norm(data, axis=1).reshape(-1, 1)
    elif method is None:
        return data


def load_data(dataset_name='DatasetA', attr_norm='l2', word_emb_norm='l2', batch_size=64):
    data_root = f'../input/{dataset_name}'
    label_list = pd.read_csv(f'{data_root}/DatasetA_train_20180813/label_list.txt', delimiter='\t', header=None)
    word_embeddings = pd.read_csv(f'{data_root}/DatasetA_train_20180813/class_wordembeddings.txt', delimiter=' ',
                                  header=None)
    attributes = pd.read_csv(f'{data_root}/DatasetA_train_20180813/attributes_per_class.txt', delimiter='\t',
                             header=None)

    tmp = label_list.merge(word_embeddings, left_on=1, right_on=0).merge(attributes, left_on='0_x', right_on=0)
    label_code = tmp['0_x']
    word_embeddings = normalize(tmp.iloc[:, 4:304].values.astype('float32'), word_emb_norm)
    attributes = normalize(tmp.iloc[:, -30:].values.astype('float32'), attr_norm)
    label2idx = {label_code[i]: i for i in range(230)}

    train_image_list = pd.read_csv(f'{data_root}/DatasetA_train_20180813/train.txt', delimiter='\t', header=None)
    train_image_list[1] = train_image_list[1].apply(lambda x: label2idx[x])

    val_image_list = pd.read_csv(f'{data_root}/DatasetA_train_20180813/submit.txt', delimiter='\t', header=None)
    val_idx = train_image_list[0].isin(val_image_list[0])

    train = train_image_list[~val_idx]
    val = train_image_list[val_idx]
    test = pd.read_csv(f'{data_root}/DatasetA_test_20180813/DatasetA_test/image.txt', header=None)

    pre_train, pre_val = train_test_split(train)

    transform_da = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(64, (0.3, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    datasets = {
        'pre_train': ZJLSet(f'{data_root}/DatasetA_train_20180813/train', pre_train[0].values, transform_da,
                            pre_train[1].values, attributes, word_embeddings),
        'pre_val': ZJLSet(f'{data_root}/DatasetA_train_20180813/train', pre_val[0].values, transform, pre_val[1].values,
                            attributes, word_embeddings),
        'train': ZJLSet(f'{data_root}/DatasetA_train_20180813/train', train[0].values, transform_da,
                        train[1].values, attributes, word_embeddings),
        'val': ZJLSet(f'{data_root}/DatasetA_train_20180813/train', val[0].values, transform, val[1].values, attributes,
                      word_embeddings),
        'all_train': ZJLSet(f'{data_root}/DatasetA_train_20180813/train', train_image_list[0].values, transform_da,
                            train_image_list[1].values, attributes, word_embeddings),
        'test': ZJLSet(f'{data_root}/DatasetA_test_20180813/DatasetA_test/test', test[0].values, transform,
                       label_code=label_code)
    }

    dataloaders = {ds: DataLoader(datasets[ds],
                                  batch_size=batch_size,
                                  shuffle=False if ds == 'test' else True,
                                  pin_memory=True,
                                  num_workers=8) for ds in datasets}

    return dataloaders, attributes, word_embeddings


def evalute(model, loader, device, word_embeddings=None):
    model = model.eval()
    preds = []
    labels = []
    for _, sample in enumerate(loader):
        image = sample['image'].to(device)
        label = sample['label'].numpy()

        if word_embeddings is None:
            pred = model.predict(image)
        else:
            pred = model.predict(image, word_embeddings)

        preds += [pred]
        labels += [label]

    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)
    return accuracy_score(labels, preds)


def train(model, device, tr_loader, n_ep, wd, *args, va_loader=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd)

    if va_loader is not None:
        max_acc = 0
        best_model_wts = copy.deepcopy(model.state_dict())

    for i_ep in range(n_ep):
        for _, sample in tqdm(enumerate(tr_loader), total=len(tr_loader)):
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            optimizer.zero_grad()
            loss = model.get_loss(image, label, *args)
            loss.backward()
            optimizer.step()

        if va_loader is not None:
            train_acc = evalute(model, va_loader, device, *args)
            val_acc = evalute(model, va_loader, device, *args)
            print(f'Ep_{i_ep} - train: {train_acc:.4%} | val: {val_acc:.4%}')

            if val_acc > max_acc:
                max_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    if va_loader is not None:
        print(f'Best - val: {max_acc:.4%}')
        model.load_state_dict(best_model_wts)

    return model


def save_result(model, device, loader, word_embeddings, path):
    print('Saving results...')
    word_embeddings = torch.from_numpy(word_embeddings).to(device)
    preds = []
    with torch.no_grad():
        for _, sample in tqdm(enumerate(loader), total=len(loader)):
            image = sample['image'].to(device)

            pred = model.predict(image, word_embeddings)
            preds += [pred]
    preds = np.concatenate(preds, 0).tolist()
    preds = loader.dataset.label_code[preds]
    result = pd.DataFrame([loader.dataset.images, preds]).T
    result.to_csv(path, sep='\t', header=None, index=None)


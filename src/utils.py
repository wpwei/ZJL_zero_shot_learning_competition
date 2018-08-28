import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ZJLSet(Dataset):
    def __init__(self, image_root, images, labels=None, attributes=None, word_embeddings=None):
        self.image_root = image_root
        self.images = images
        self.labels = labels
        self.attributes = attributes
        self.word_embeddings = word_embeddings
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

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


def load_data(dataset_name='DatasetA', attr_norm='l2', word_emb_norm='l2', batch_size=128):
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

    datasets = {
        'train': ZJLSet(f'{data_root}/DatasetA_train_20180813/train', train[0].values, train[1].values, attributes,
                        word_embeddings),
        'val': ZJLSet(f'{data_root}/DatasetA_train_20180813/train', val[0].values, val[1].values, attributes,
                      word_embeddings),
        'test': ZJLSet(f'{data_root}/DatasetA_test_20180813/DatasetA_test/test', test[0].values)
    }

    dataloaders = {ds: DataLoader(datasets[ds],
                                  batch_size=batch_size,
                                  shuffle=False if ds == 'test' else True,
                                  pin_memory=True,
                                  num_workers=8) for ds in datasets}

    return dataloaders, attributes, word_embeddings

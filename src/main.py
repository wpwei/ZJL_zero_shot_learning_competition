from src.utils import load_data, save_result, train
from src.models import DEM
from src.densenet import DenseNet
import torch
import argparse


def pre_train_cnn(cnn, device, loaders, n_ep=200, val=False, wd=1e-4, lr=1e-3, save_model_path=None):
    print('Training CNN...')

    model = cnn(num_classes=230)

    if val:
        tr_loader = loaders['pre_train']
        va_loader = loaders['pre_val']
    else:
        tr_loader = loaders['all_train']
        va_loader = None

    model = train(model, device, tr_loader, n_ep, wd, lr, va_loader=va_loader)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    return model


def train_DEM(cnn, device, loaders, word_embeddings, n_ep=50, wd=1e-8, lr=1e-4, val=False, save_model_path=None):
    print('Training DEM...')

    model = DEM(cnn)
    word_embeddings = torch.from_numpy(word_embeddings).to(device)

    if val:
        tr_loader = loaders['train']
        va_loader = loaders['val']
    else:
        tr_loader = loaders['all_train']
        va_loader = None

    model = train(model, device, tr_loader, n_ep, wd, lr, word_embeddings, va_loader=va_loader, grad_norm_clip=1)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=None,
                        help='GPU ID to use, e.g. \'0\'', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    gpu_id = args.gpu_id
    if gpu_id is None:
        DEVICE = torch.device(f'cpu')
    else:
        DEVICE = torch.device(f'cuda:{gpu_id}')

    loaders, attributes, word_embeddings = load_data()

    cnn = pre_train_cnn(DenseNet, DEVICE, loaders, n_ep=200, save_model_path='../output/densenet.pkl')

    model = train_DEM(cnn, DEVICE, loaders, word_embeddings, n_ep=5, save_model_path='../output/dem.pkl')

    save_result(model, DEVICE, loaders['test'], word_embeddings, '../output/result.txt')

from src.utils import load_data, evalute, save_result, train
from src.DeViSE import DeViSE
from src.densenet import DenseNet
import torch


def pre_train_cnn(cnn, device, loaders, n_ep=200, val=False, wd=1e-4, save_model_path=None):
    print('Training CNN...')

    model = cnn(num_classes=230)

    if val:
        tr_loader = loaders['pre_train']
        va_loader = loaders['pre_val']
    else:
        # tr_loader = loaders['all_train']
        tr_loader = loaders['train']
        va_loader = None

    model = train(model, device, tr_loader, n_ep, wd, va_loader=va_loader)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    return model


def train_DeViSE(cnn, device, loaders, word_embeddings, n_ep=100, wd=1e-4, val=False):
    print('Training DeViSE...')

    model = DeViSE(cnn)
    word_embeddings = torch.from_numpy(word_embeddings).to(device)

    if val:
        tr_loader = loaders['train']
        va_loader = loaders['val']
    else:
        tr_loader = loaders['all_train']
        va_loader = None

    model = train(model, device, tr_loader, n_ep, wd, word_embeddings, va_loader=va_loader)

    return model


if __name__ == '__main__':
    DEVICE = torch.device('cuda:2')
    loaders, attributes, word_embeddings = load_data()

    cnn = pre_train_cnn(DenseNet, DEVICE, loaders, n_ep=200, save_model_path='../output/densenet.pkl')
    devise = train_DeViSE(cnn, DEVICE, loaders, word_embeddings, n_ep=100, val=True)
    save_result(devise, DEVICE, loaders['test'], word_embeddings, '../output/result.txt')

from src.utils import load_data, evalute
from src.DeViSE import SimpleCNN, DeViSE
import torch
import copy


def pre_train_cnn(cnn, device, loaders, n_ep=500):
    model = cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    max_acc = 0
    for i_ep in range(n_ep):
        model = model.train()
        for _, sample in enumerate(loaders['pre_train']):
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            optimizer.zero_grad()
            loss = model.get_loss(image, label)
            loss.backward()
            optimizer.step()

        train_acc = evalute(model, loaders['pre_train'], device)
        val_acc = evalute(model, loaders['pre_val'], device)
        print(f'Ep_{i_ep} - train: {train_acc:.4%} | val: {val_acc:.4%}')

        if val_acc > max_acc:
            max_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best - val: {max_acc:.4%}')
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    DEVICE = torch.device('cuda:2')
    loaders, attributes, word_embeddings = load_data()

    cnn = pre_train_cnn(SimpleCNN, DEVICE, loaders)
    #
    # cnn = SimpleCNN().to(DEVICE)
    # model = cnn
    # # model = DeViSE().to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    # word_embeddings = torch.from_numpy(word_embeddings).to(DEVICE)
    #
    # for i in range(100):
    #     model = model.train()
    #     for _, sample in enumerate(loaders['pre_train']):
    #         image = sample['image'].to(DEVICE)
    #         label = sample['label'].to(DEVICE)
    #
    #         optimizer.zero_grad()
    #         loss = model.get_loss(image, label, word_embeddings)
    #         loss.backward()
    #         optimizer.step()
    #
    #     train_acc = evalute(model, word_embeddings, loaders['pre_train'], DEVICE)
    #     val_acc = evalute(model, word_embeddings, loaders['pre_val'], DEVICE)
    #     print(f'Ep_{i} - train: {train_acc:.4%} | val: {val_acc:.4%}')
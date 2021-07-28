import torch
import torch.nn as nn
import pandas as pd
from overall import FILENAME, LABELS, BATCH_SIZE
from overall import test_percent
from overall import CUDA, DEVICE, EPOCHS
from overall import save_model_name, LR
from overall import NUM_CLASS
from datapreprocessing import get_dataloader
from model_deepcnn import DPCNN
from sae import SAE
from evaluate import evaluate_loss_acc
import time

train_loader, test_loader = get_dataloader(
    FILENAME, LABELS,
    test_percent, BATCH_SIZE)


def deal_matrix(matrix):
    row_sum = matrix.sum(axis=1)  # precision
    col_sum = matrix.sum(axis=0)  # recall
    P, R = [], []
    n_class = matrix.shape[0]
    for i in range(n_class):
        p = matrix[i][i] / row_sum[i]
        r = matrix[i][i] / col_sum[i]
        P.append(p)
        R.append(r)
    print('Precison: ')
    print(P)
    print('Recall: ')
    print(R)


# model
model = DPCNN()
# model = SAE()
# model.load_state_dict(torch.load(load_model_name))

if CUDA:
    model = model.cuda(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=LR)
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss()

num_batch = len(train_loader)
print('data prepare ok! total ',
      num_batch, ' batch for 1 epoch.')

plot_train_loss = []
plot_train_accuracy = []

best_test_acc_reports = [0., None]

# train
for epoch in range(EPOCHS):
    train_start = time.time()
    for i in range(num_batch):
        batch_X, batch_y = train_loader[i]
        if batch_X.size(0) == 0:
            print('encounter batch with size 0!')
            continue
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
        # print(batch_X.size())
        # begin to train
        model.train()
        optimizer.zero_grad()
        out = model(batch_X)

        loss = loss_func(out, batch_y.long())
        loss.backward()
        optimizer.step()

        # if i % 80 == 0:
        #     #evaluate train
        #     train_loss, train_accuracy, train_cmatrix = evaluate_loss_acc(model, train_loader, overall_label_ix, is_cm = False)
        #     test_loss, test_accuracy, test_cmatrix = evaluate_loss_acc(model, test_loader, overall_label_ix, is_cm = False)
        #     print('epoch: {}, batch: {}, train loss: {}, train accuracy: {}, test loss: {}, test accuracy: {}'.format(epoch, i, train_loss, train_accuracy, test_loss, test_accuracy))
        #     plot_train_loss.append(train_loss)
        #     plot_train_accuracy.append(train_accuracy)
        #     plot_test_loss.append(test_loss)
        #     plot_test_accuracy.append(test_accuracy)
        if i % 3000 == 0:
            print('batch {} ok'.format(i))
            # train_loss, train_accuracy, train_cmatrix = evaluate_loss_acc(model, train_loader, is_cm = False)
            # print('train loss: {}, train accuracy: {}'.format(train_loss, train_accuracy))
            # plot_train_loss.append(train_loss)
            # plot_train_accuracy.append(train_accuracy)

    train_loss, train_accuracy, train_cmatrix, _ = evaluate_loss_acc(
        model, train_loader, is_cm=False)
    test_loss, test_accuracy, test_cmatrix, results_report = evaluate_loss_acc(
        model, test_loader, is_cm=True)
    if best_test_acc_reports[0] < test_accuracy:
        best_test_acc_reports[0] = test_accuracy
        best_test_acc_reports[1] = results_report
    deal_matrix(test_cmatrix)
    print('epoch: {} ok,  train loss: {}, train accuracy: {}'
          ', test loss: {}, test accuracy: {}'.format(
        epoch, train_loss, train_accuracy, test_loss, test_accuracy))
    # print('epoch: {} ok,  train loss: {}, train accuracy: {}'.format(epoch, train_loss, train_accuracy))
    # plot_train_loss.append(train_loss)
    # plot_train_accuracy.append(train_accuracy)

print('#'*30, '\nThe best test acc:', best_test_acc_reports[0], '\n', '#'*30)
df = pd.DataFrame({
    'App': LABELS, 'Precision': [best_test_acc_reports[1][i][
        'precision'] for i in range(0, NUM_CLASS)],
    'Recall': [best_test_acc_reports[1][i][
        'recall'] for i in range(0, NUM_CLASS)],
    'F1 score': [best_test_acc_reports[1][i][
        'f1-score'] for i in range(0, NUM_CLASS)]})
df.to_excel('deeppacket_' + model.__name__ +
            '_dataset_new20_results.xlsx')

# save model
torch.save(model.state_dict(), save_model_name)

# save loss and accuracy

# L = [plot_train_loss, plot_train_accuracy, plot_test_loss, plot_test_accuracy]
# with open('loss_accuracy', 'wb') as f:
#     pickle.dump(L, f)

# with open('cmatrix', 'wb') as f:
#     pickle.dump(plot_cmatrix, f)

# with open('figure/train_loss', 'wb') as f:
#     pickle.dump(plot_train_loss, f)

# with open('figure/train_accuracy', 'wb') as f:
#     pickle.dump(plot_train_accuracy, f)

print('ok')

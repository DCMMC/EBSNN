from sklearn.metrics import classification_report
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from overall import CUDA, DEVICE, NUM_CLASS
from time import time


def evaluate_loss_acc(model, dataloader, is_cm):
    loss_func = nn.CrossEntropyLoss()
    total_loss = 0
    y_hat, y = [], []
    time_logs = []

    for i in range(len(dataloader)):
        batch_X, batch_y = dataloader[i]
        if batch_X.size(0) == 0:
            print('encountering batch with size 0 in evaluating')
            continue
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)

        s_t = time()
        model.eval()
        out = model(batch_X)
        time_logs.append(time() - s_t)

        loss = loss_func(out, batch_y.long())
        total_loss += loss.item()
        # ----------------

        y_hat += out.max(1)[1].tolist()
        y += batch_y.tolist()

    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    cmatrix = 0
    if is_cm:
        cmatrix = confusion_matrix(
            y, y_hat,
            labels=list(range(0, NUM_CLASS)))
       # cmatrix = confusion_matrix(y, y_hat, labels=[0,1,2])
    results_report = classification_report(
        y, y_hat, labels=list(range(0, NUM_CLASS)),
        output_dict=True)
    print('DEBUG inference time per batch:',
          str(sum(time_logs) / len(time_logs)))


    return total_loss, accuracy, cmatrix, results_report

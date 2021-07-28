import sys
import os
from shutil import copyfile
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

p_log_file = 'log_20/log_train_flow_K3_gamma1_B128_E64.txt'


def set_log_file(log_filename):
    global p_log_file
    p_log_file = log_filename
    if os.path.exists(p_log_file):
        print('WARNING: file {} already exists, make a backup.'.format(
            p_log_file))
        copyfile(p_log_file, p_log_file + '.bak')
    print('INFO: log will write into {}.'.format(p_log_file))
    p_log_file = open(p_log_file, 'w')


def p_log(*ks, **kwargs):
    print(*ks, **kwargs)
    sys.stdout.flush()
    stdout = sys.stdout
    sys.stdout = p_log_file
    print(*ks, **kwargs)
    sys.stdout.flush()
    sys.stdout = stdout


def deal_results(y_true, y_pred, digits=4):
    p_log('Confison Matrix:\n', confusion_matrix(y_true, y_pred))
    p_log(classification_report(y_true, y_pred, digits=digits))
    return classification_report(y_true, y_pred,
                                 output_dict=True,
                                 digits=digits)

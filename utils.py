import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap(name='Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.draw()   



def save_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap(name='Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("{0}.png".format(title), dpi=300)

dic_tot = {}

def one_hot_encode(labels):
    """
    Convert labels to one-hot matrix. Each row is a label, each column is a unique label.
    :param labels:
    :return: one-hot-vector matrix for the labels
    """
    n_labels = len(labels)
    # n_unique_labels = len(np.unique(labels))
    n_unique_labels = 56
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    # print (labels)
    # print (n_labels)
    # print (np.arange(n_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def transform_labels_numbers(labels_s):
    new_labels = []
    for l in labels_s:
        new_labels.append(dic_tot[l])
    return new_labels


def from_plus_to_one_hot(pred_labels):
    final_conv_labels = []
    list_pl = list(pred_labels)
    for l in list_pl:
        l_list = list(l)
        temp_index = [i for i, j in enumerate(l_list) if j == 1]
        if len(temp_index) == 1:
            final_conv_labels.append(temp_index[0] + 1)
        if len(temp_index) == 0:
            final_conv_labels.append(1000000)
        if len(temp_index) > 1:
            t_1 = temp_index[0] + 1
            t_2 = temp_index[1] + 1
            final_conv_labels.append(int(str(t_1) + str(t_2)))
    final_conv_labels_1hot = transform_labels_numbers(final_conv_labels)
    # for f in final_conv_labels:
    #   final_conv_labels_1hot.append(f - 1)
    final_conv_labels_1hot = one_hot_encode(final_conv_labels_1hot)
    return final_conv_labels_1hot

classes = ['ac', 'ch', 'cp', 'db', 'd', 'ei', 'gs', 'j', 's', 'sm',
           'ac+ch', 'ac+cp', 'ac+db',
           'ac+d', 'ac+ei', 'ac+gs',
           'ac+j', 'ac+s', 'ac+sm',
           'ch+cp',
           'ch+db', 'ch+d', 'ch+ei', 'ch+gs',
           'ch+j',
           'ch+s', 'ch+sm', 'cp+db', 'cp+d',
           'cp+ei', 'cp+gs', 'cp+j',
           'cp+s', 'cp+sm', 'db+d', 'db+ei',
           'db+gs', 'db+j', 'db+s', 'db+sm',
           'd+ei', 'd+gs', 'd+j', 'd+s',
           'd+sm', 'ei+gs', 'ei+j', 'ei+s',
           'ei+sm', 'gs+j', 'gs+s', 'gs+sm',
           'j+s', 'j+sm', 's+sm', 'xxx']

single_classes = ['ac', 'ch', 'cp', 'db', 'd', 'ei', 'gs', 'j', 's', 'sm']
combined_classes = ['ac+ch', 'ac+cp', 'ac+db',
                    'ac+d', 'ac+ei', 'ac+gs',
                    'ac+j', 'ac+s', 'ac+sm',
                    'ch+cp',
                    'ch+db', 'ch+d', 'ch+ei', 'ch+gs',
                    'ch+j',
                    'ch+s', 'ch+sm', 'cp+db', 'cp+d',
                    'cp+ei', 'cp+gs', 'cp+j',
                    'cp+s', 'cp+sm', 'db+d', 'db+ei',
                    'db+gs', 'db+j', 'db+s', 'db+sm',
                    'd+ei', 'd+gs', 'd+j', 'd+s',
                    'd+sm', 'ei+gs', 'ei+j', 'ei+s',
                    'ei+sm', 'gs+j', 'gs+s', 'gs+sm',
                    'j+s', 'j+sm', 's+sm']

def classes_number_mapper():
    dic_single_classes = {}
    # dic_tot = {}
    x = 0
    for s in single_classes:
        dic_single_classes[s] = x + 1
        x += 1
    for n in range(0, 10):
        dic_tot[n + 1] = n
    x = 10
    for s in combined_classes:
        splitted_s = s.split("+")
        t1 = dic_single_classes[splitted_s[0]]
        t2 = dic_single_classes[splitted_s[1]]
        t_comb = int(str(t1) + str(t2))
        dic_tot[t_comb] = x
        x += 1
    dic_tot[1000000] = x
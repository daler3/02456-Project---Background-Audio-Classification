from preprocessor import preprocessor
from sklearn import metrics
from keras.models import load_model
import numpy as np
import utils

classes = classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
load_path = 'extracted/long_60'
train_dirs = []
n_folders = 10
for i in range(1, n_folders + 1):
    train_dirs.append('fold{0}'.format(i))
val_fold = 'fold9'
test_fold= 'fold10'
train_dirs.remove(val_fold)
train_dirs.remove(test_fold)

# print("Run {0}: test folder is fold{0}".format(folds[1]) + ", validation folder is fold{0}".format(folds[0]))
pp = preprocessor()
pp.load_extracted_fts_lbs(load_path=load_path, train_dirs=train_dirs, test_fold=test_fold, val_fold=val_fold)
train_dirs.append(val_fold)
train_dirs.append(test_fold)
print("Data prep completed")
model = load_model('models/long60_150_(9, 10).h5')    
preds = model.predict_classes(pp.test_x, verbose=0)
confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
utils.save_confusion_matrix(confusion_matrix, classes, title='long60_150_(9, 10)')

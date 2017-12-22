from keras.models import load_model
from preprocessor import preprocessor
from sklearn import metrics, model_selection
import utils
import numpy as np

def plot_confusion_matrix(model_filename, load_path, save=False):
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer', 'siren', 'street_music']

    model = load_model(model_filename)
    # example: long200_150_(1,2).h5
    test_fold = 'fold' + model_filename.split(', ')[1].split(')')[0]

    pp = preprocessor()
    pp.load_extracted_fts_lbs(train_dirs=[test_fold], load_path=load_path)
    preds = model.predict_classes(pp.train_x, verbose=0)
    # write_preds(preds, output_predictions_file)
    cm = metrics.confusion_matrix(np.argmax(pp.train_y, axis=1), preds)
    if save:
        utils.save_confusion_matrix(cm, classes)
    else:
        utils.plot_confusion_matrix(cm, classes)
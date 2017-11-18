from preprocessor import preprocessor
from keras_models import piczak_CNN
from keras.callbacks import TensorBoard
from sklearn import metrics
import numpy as np
import pandas as pd
import utils

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
train_dirs = []
save_dir = 'C:\\Deep Learning Dataset\\UrbanSound8K\\extracted_long'
for i in range(2,11):
    train_dirs.append('fold{0}'.format(i))

def train_keras_cnn(epochs=100, output_model_file="./piczak_model_fold1_only.h5",
                    output_predictions_file="./test.csv"):

    pp = preprocessor(parent_dir='C:\\Deep Learning Dataset\\UrbanSound8K\\audio')
    print("Loading the data...")
    # Run this to load the extracted features
    # pp.data_prep(train_dirs=["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"], load_path='extracted')
    pp.data_prep(train_dirs=train_dirs, load_path=save_dir)

    tb = TensorBoard(log_dir='./TensorBoard')

    cvscores = []
    for i in range(1, 11):
        train_dirs.remove('fold{0}'.format(i))
        test_dir='fold{0}'.format(i)
        print("Run {0}: test folder is fold{0}").format(i)

        pp.data_prep(train_dirs=train_dirs, load_path='extracted', test_fold=test_dir)
        train_dirs.append('fold{0}'.format(i))

        model = piczak_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])

        model.fit(pp.train_x, pp.train_y, epochs=epochs,
                  batch_size=256, verbose=0, callbacks=[tb])

        #print("Saving the model to file: {0}".format(output_model_file))
        #model.save(output_model_file)

        scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
        print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

        #print("Writing test predictions to csv file : {0}".format(output_predictions_file))
        #def write_preds(preds, fname):
        #    pd.DataFrame({"Predictions": preds, "Actual": np.argmax(pp.test_y, axis=1)}).to_csv(fname, index=False,
        #                                                                                      header=True)
        cvscores.append(scores[1] * 100)
        #preds = model.predict_classes(pp.test_x, verbose=0)
        #write_preds(preds, output_predictions_file)
        #confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
        #utils.plot_confusion_matrix(confusion_matrix, classes)

    print("Average performance after cross-validation: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


if __name__ == '__main__':
    train_keras_cnn()

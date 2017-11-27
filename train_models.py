from preprocessor import preprocessor
from keras_models import piczak_CNN
from keras.callbacks import TensorBoard
from sklearn import metrics
import numpy as np
import pandas as pd
import logging

def train_keras_cnn(epochs):
    logging.basicConfig(filename='cv.log', filemode='w', level=logging.DEBUG)
    cvscores = []

    for i in range(1, n_folders + 1):
        train_dirs.remove('fold{0}'.format(i))
        test_dir='fold{0}'.format(i)

        logging.info("Run {0}: test folder is fold{0}".format(i))
        # tb = TensorBoard(log_dir='./TensorBoard/' + 'run{0}'.format(i))
        # tb = TensorBoard(log_dir='./TensorBoard/' + 'run{0}'.format(i), histogram_freq=1, write_graph=True,
        #                                         write_images=True)

        pp = preprocessor(parent_dir='data/UrbanSound8K/audio')
        pp.data_prep(train_dirs=train_dirs, test_fold=test_dir, load_path='extracted')
        logging.info("Data prep completed")
        train_dirs.append('fold{0}'.format(i))

        model = piczak_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])
        logging.info("Model built")

        model.fit(pp.train_x, pp.train_y, validation_data=(pp.test_x, pp.test_y), epochs=epochs,
                  batch_size=256, verbose=2)
        logging.info("Model trained")

        output_model_file = 'models/long_' + str(epochs) + '_' + str(i) + '.h5'
        logging.info("Saving the model to file: {0}".format(output_model_file))
        model.save(output_model_file)

        scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
        logging.info("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

        #logging.info("Writing test predictions to csv file : {0}".format(output_predictions_file))
        #def write_preds(preds, fname):
        #    pd.DataFrame({"Predictions": preds, "Actual": np.argmax(pp.test_y, axis=1)}).to_csv(fname, index=False,
        #                                                                                      header=True)
        cvscores.append(scores[1] * 100)
        #preds = model.predict_classes(pp.test_x, verbose=0)
        #write_preds(preds, output_predictions_file)
        #confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
        #utils.plot_confusion_matrix(confusion_matrix, classes)

    logging.info("Average performance after cross-validation: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == '__main__':
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer', 'siren', 'street_music']
    train_dirs = []

    n_folders = 10
    for i in range(1, n_folders + 1):
        train_dirs.append('fold{0}'.format(i))

    train_keras_cnn(epochs=150)

from preprocessor import preprocessor
from keras_models import piczak_CNN
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn import metrics, model_selection
import numpy as np
import pandas as pd
import logging
from keras import optimizers


def piczac_cross_validation(epochs, load_path):
    cvscores = []
    for folds in ((10, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)):
        val_fold = 'fold' + str(folds[0])
        test_fold= 'fold' + str(folds[1])
        train_dirs.remove(val_fold)
        train_dirs.remove(test_fold)

        print("Run {0}: test folder is fold{0}".format(folds[1]) + ", validation folder is fold{0}".format(folds[0]))

        tb = TensorBoard(log_dir='./TensorBoard/' + 'run{0}'.format(folds[1]))
        es = EarlyStopping(patience=10, verbose=1)

        pp = preprocessor()
        pp.load_extracted_fts_lbs(load_path=load_path, train_dirs=train_dirs, test_fold=test_fold, val_fold=val_fold)
        train_dirs.append(val_fold)
        train_dirs.append(test_fold)
        print("Data prep completed")

        model = piczak_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])
        print("Model built")

        model.fit(pp.train_x, pp.train_y, validation_data=[pp.val_x, pp.val_y], epochs=epochs,
                  batch_size=1000, verbose=2, callbacks=[tb, es])
        print("Model trained")

        #output_model_file = 'models/long_' + str(epochs) + '_' + str(folds) + '.h5'
        #logging.info("Saving the model to file: {0}".format(output_model_file))
        #model.save(output_model_file)

        scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
        print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))
        #logging.info("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

        #logging.info("Writing test predictions to csv file : {0}".format(output_predictions_file))
        #def write_preds(preds, fname):
        #    pd.DataFrame({"Predictions": preds, "Actual": np.argmax(pp.test_y, axis=1)}).to_csv(fname, index=False,
        #                                                                                      header=True)
        cvscores.append(scores[1] * 100)
        #preds = model.predict_classes(pp.test_x, verbose=0)
        #write_preds(preds, output_predictions_file)
        #confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
        #utils.plot_confusion_matrix(confusion_matrix, classes)
    print("Average performance after cross-validation: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def scikit_cross_validation(epochs, load_path):
    pp = preprocessor()
    pp.load_extracted_fts_lbs(train_dirs=train_dirs,
                 load_path=load_path)  # test_fold=test_dir,
    logging.info("Data prep completed")

    CV = model_selection.KFold(10, shuffle=True)

    k = 0
    es = EarlyStopping(patience=5, verbose=1)
    for train_index, test_index in CV.split(pp.X):
        print('Computing CV fold: {0}/{1}..'.format(k + 1, 10))
        tb = TensorBoard(log_dir='./TensorBoard/' + 'run{0}'.format(k + 1))
        # extract training and test set for current CV fold
        X_train, y_train = pp.X[train_index, :], pp.y[train_index]
        X_test, y_test = pp.X[test_index, :], pp.y[test_index]

        model = piczak_CNN(input_dim=X_train[0].shape, output_dim=y_train.shape[1])
        logging.info("Model built")

        model.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
                  batch_size=1000, verbose=2, callbacks=[tb])
        logging.info("Model trained")

        scores = model.evaluate(X_test, y_test, verbose=2)
        # logging.info("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))
        print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))
        k = k + 1


if __name__ == '__main__':
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer', 'siren', 'street_music']
    train_dirs = []

    #logging.basicConfig(filename='cv.log', filemode='w', level=logging.DEBUG)

    n_folders = 10
    for i in range(1, n_folders + 1):
        train_dirs.append('fold{0}'.format(i))

    # if using long segments, use 150 epochs. if using short, use 300
    # piczac_cross_validation(epochs=150, load_path='extracted_short_200')
    scikit_cross_validation(epochs=150, load_path='extracted_short_200')

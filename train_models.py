from preprocessor import preprocessor
from keras_models import piczak_CNN
from keras.callbacks import TensorBoard, EarlyStopping
from keras import optimizers
from sklearn import metrics, model_selection
import numpy as np
import pandas as pd
import logging

def train_keras_cnn(epochs):
    logging.basicConfig(filename='cv.log', filemode='w', level=logging.DEBUG)
    cvscores = []

    # # for i in range(1, n_folders + 1):
    #     train_dirs.remove('fold{0}'.format(i))
    #     test_dir='fold{0}'.format(i)

    # logging.info("Run {0}: test folder is fold{0}".format(i))
    # tb = TensorBoard(log_dir='./TensorBoard/' + 'run{0}'.format(i), histogram_freq=1, write_graph=True,
    #                                         write_images=True)

    pp = preprocessor(parent_dir='C:\\Deep Learning Dataset\\UrbanSound8K\\audio')
    pp.data_prep(train_dirs=train_dirs, load_path='C:\\Deep Learning Dataset\\UrbanSound8K\\extracted_long') #test_fold=test_dir,
    logging.info("Data prep completed")
    # train_dirs.append('fold{0}'.format(i))
    K=10
    CV = model_selection.KFold(K,shuffle=True)
    sgd = optimizers.SGD(lr=0.002, nesterov=True, momentum=0.9, decay=0.001)
    k=0
    es = EarlyStopping(patience=5,verbose=1)
    for train_index, test_index in CV.split(pp.X):
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))
        tb = TensorBoard(log_dir='./TensorBoard/' + 'run{0}'.format(k+1))
        # extract training and test set for current CV fold
        X_train, y_train = pp.X[train_index,:], pp.y[train_index]
        X_test, y_test = pp.X[test_index,:], pp.y[test_index]

        model = piczak_CNN(input_dim=X_train[0].shape, output_dim=y_train.shape[1], optimizer=sgd)
        logging.info("Model built")

        model.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
                    batch_size=1000, verbose=2 , callbacks=[tb,es])
        logging.info("Model trained")

        scores = model.evaluate(X_test, y_test, verbose=2)
        # logging.info("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))
        print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))
        k = k+1  

    # output_model_file = 'models/long_' + str(epochs) + '_' + str(i) + '.h5'
    # logging.info("Saving the model to file: {0}".format(output_model_file))
    # model.save(output_model_file)


    #logging.info("Writing test predictions to csv file : {0}".format(output_predictions_file))
    #def write_preds(preds, fname):
    #    pd.DataFrame({"Predictions": preds, "Actual": np.argmax(pp.test_y, axis=1)}).to_csv(fname, index=False,
    #                                                                                      header=True)
    # cvscores.append(scores[1] * 100)
    #preds = model.predict_classes(pp.test_x, verbose=0)
    #write_preds(preds, output_predictions_file)
    #confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
    #utils.plot_confusion_matrix(confusion_matrix, classes)

    # logging.info("Average performance after cross-validation: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == '__main__':
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer', 'siren', 'street_music']
    train_dirs = []

    n_folders = 10
    for i in range(1, n_folders + 1):
        train_dirs.append('fold{0}'.format(i))

    train_keras_cnn(epochs=25)

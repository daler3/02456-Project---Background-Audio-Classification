from preprocessor import preprocessor
# from keras_models import piczak_new_CNN
from keras.callbacks import TensorBoard, EarlyStopping
# from sklearn import metrics, model_selection
# import numpy as np
# import pandas as pd
# import logging
# from keras import optimizers
from keras.models import load_model
from keras import backend as K

volumeOverlay = 30

train_dirs = []

    #logging.basicConfig(filename='cv.log', filemode='w', level=logging.DEBUG)

n_folders = 10
for i in range(1, n_folders + 1):
    train_dirs.append('overlap/fold{0}_overlap_{1}dB'.format(i, volumeOverlay))

def piczac_cross_validation(epochs, load_path):
    
    for fold in ((10, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)):
        val_fold = 'overlap/fold{0}_overlap_{1}dB'.format(fold[0],volumeOverlay)
        test_fold= 'overlap/fold{0}_overlap_{1}dB'.format(fold[1],volumeOverlay)
        single_fold = 'fold{0}'.format(fold[1]) 
        train_dirs.remove(val_fold)
        train_dirs.remove(test_fold)
        model = load_model('Models/short60_300_{0}.h5'.format(str(fold)))
        pp = preprocessor(parent_dir='C:\\Deep Learning Dataset\\UrbanSound8K\\audio_overlap')
        pp.load_extracted_fts_lbs(load_path=load_path, train_dirs=train_dirs, val_fold=val_fold, test_fold=test_fold, single_fold=single_fold)        

        scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
        print("Test_fold: {2} Pretrain overlap - loss: {0}, test-acc: {1}".format(scores[0], scores[1], fold[1]))
        scores = model.evaluate(pp.single_x, pp.single_y, verbose=0)        
        print("Test_fold: {2} Pretrain single - loss: {0}, test-acc: {1}".format(scores[0], scores[1], fold[1]))
        
        tb = TensorBoard(log_dir='./TensorBoard/' + 'overlap_run{0}'.format(fold[1]))
        es = EarlyStopping(patience=10, verbose=1)

        model.fit(pp.train_x, pp.train_y, validation_data=[pp.val_x, pp.val_y], epochs=epochs,
                  batch_size=1000, verbose=0, callbacks=[tb, es])
        scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
        print("Test_fold: {2} Posttrain - loss: {0}, test-acc: {1}".format(scores[0], scores[1], fold[1]))
        scores = model.evaluate(pp.single_x, pp.single_y, verbose=0)        
        print("Test_fold: {2} Posttrain single - loss: {0}, test-acc: {1}".format(scores[0], scores[1], fold[1]))
        K.clear_session()
        
        
        train_dirs.append(val_fold)
        train_dirs.append(test_fold)
        # val_fold = 'fold' + str(folds[0])
        # test_fold= 'fold' + str(folds[1])


        # print("Run {0}: test folder is fold{0}".format(folds[1]) + ", validation folder is fold{0}".format(folds[0]))


        # pp = preprocessor()
        # pp.load_extracted_fts_lbs(load_path=load_path, train_dirs=train_dirs, test_fold=test_fold, val_fold=val_fold)
        # print("Data prep completed")

        # model = piczak_new_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])
        # print("Model built")
        # print("Model trained")

        # #output_model_file = 'models/long_' + str(epochs) + '_' + str(folds) + '.h5'
        # #logging.info("Saving the model to file: {0}".format(output_model_file))
        # #model.save(output_model_file)

        # scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
        # print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))
        # #logging.info("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

        # #logging.info("Writing test predictions to csv file : {0}".format(output_predictions_file))
        # #def write_preds(preds, fname):
        # #    pd.DataFrame({"Predictions": preds, "Actual": np.argmax(pp.test_y, axis=1)}).to_csv(fname, index=False,
        # #                                                                                      header=True)
        # cvscores.append(scores[1] * 100)
        #preds = model.predict_classes(pp.test_x, verbose=0)
        #write_preds(preds, output_predictions_file)
        #confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
        #utils.plot_confusion_matrix(confusion_matrix, classes)
    # print("Average performance after cross-validation: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == '__main__':
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer', 'siren', 'street_music']
    # train_dirs = []

    #logging.basicConfig(filename='cv.log', filemode='w', level=logging.DEBUG)

    # n_folders = 2
    # for i in range(2, n_folders + 1):
    #     train_dirs.append('fold{0}_overlap_{1}dB'.format(i,volumeOverlay))
    # n_folders = 1
    # for i in range(1, n_folders + 1):
    #     train_dirs.append('fold{0}'.format(i))

    # if using long segments, use 150 epochs. if using short, use 300
    piczac_cross_validation(epochs=25, load_path='C:\\Deep Learning Dataset\\UrbanSound8K\\extracted_short_60')
    # scikit_cross_validation(epochs=150, load_path='extracted_short_200')

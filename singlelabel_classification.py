from preprocessor import preprocessor
from keras_models import piczak_CNN
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
from keras import backend as K
import utils
classes = classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
def piczac_cross_validation(epochs, load_path):
    train_dirs = []

    n_folders = 10
    for i in range(1, n_folders + 1):
        train_dirs.append('fold{0}'.format(i))

    cvscores = []  
    for folds in [(9, 10)]:
        val_fold = 'fold' + str(folds[0])
        test_fold= 'fold' + str(folds[1])
        # Remove validation and test from train
        train_dirs.remove(val_fold)
        train_dirs.remove(test_fold)

        print("Run {0}: test folder is fold{0}".format(folds[1]) + ", validation folder is fold{0}".format(folds[0]))

        # tb = TensorBoard(log_dir='./TensorBoard/short_60/' + 'run{0}'.format(folds[1]))
        es = EarlyStopping(patience=10, verbose=1)

        pp = preprocessor()
        pp.load_extracted_fts_lbs(load_path=load_path, train_dirs=train_dirs, test_fold=test_fold, val_fold=val_fold)
        train_dirs.append(val_fold)
        train_dirs.append(test_fold)
        print("Data prep completed")

        model = piczak_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])
        print("Model built")

        model.fit(pp.train_x, pp.train_y, validation_data=[pp.val_x, pp.val_y], epochs=epochs,
                   batch_size=1000, verbose=2, callbacks=[es])
        print("Model trained")

        output_model_file = 'models/long60_' + str(epochs) + '_' + str(folds) + '.h5'
        model.save(output_model_file)
        scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
        print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))                                                                     
        cvscores.append(scores[1] * 100)      
        K.clear_session()        

    print("Average performance after cross-validation: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == '__main__':
    # if using long segments, use 150 epochs. if using short, use 300
    # change tensorboard folder and model output file

    piczac_cross_validation(epochs=150, load_path="E:\\Deep Learning Datasets\\UrbanSound8K\\extracted_long")
    #piczac_cross_validation(epochs=300, load_path='extracted_short_60')
    #model_filename = 'models/long60/long60_150_(9, 10).h5'
    #load_path = 'data/extracted_long_60'
    #plot_confusion_matrix(model_filename, load_path)
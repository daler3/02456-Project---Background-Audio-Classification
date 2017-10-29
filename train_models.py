from preprocessor import preprocessor
from keras_models import piczak_CNN
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd


def train_keras_cnn(epochs=100, output_model_file="./piczak_model.h5",
                    output_predictions_file="./test.csv", save_data_to_csv=False):
    pp = preprocessor(parent_dir='data/UrbanSound8K/audio')
    pp.data_prep(sub_dirs=['fold1'])

    tb = TensorBoard(log_dir='./TensorBoard')

    print("Building the model...")
    train_data = pp.train
    model = piczak_CNN(input_dim=train_data[0][0].shape, output_dim=10)

    print("Training the model...")
    model.fit(train_data[0], train_data[1], epochs=epochs,
              batch_size=256, validation_split=0.2, verbose=2, callbacks=[tb])

    print("Saving the model to file: {0}".format(output_model_file))
    model.save(output_model_file)

    print("Evaluating test performance...")
    test_data = pp.test
    performance = model.evaluate(test_data[0], test_data[1], verbose=0)
    print("loss: {0}, test-acc: {1}".format(performance[0], performance[1]))

    print("Writing test predictions to csv file : {0}".format(output_predictions_file))
    def write_preds(preds, fname):
        pd.DataFrame({"Predictions": preds, "Actual": np.argmax(test_data[1], axis=1)}).to_csv(fname, index=False,
                                                                                          header=True)

    preds = model.predict_classes(test_data[0], verbose=0)
    write_preds(preds, output_predictions_file)


if __name__ == '__main__':
    train_keras_cnn(epochs=250,
                      save_data_to_csv=True,
                      output_model_file='trained_models/250-epoch-cnn.h5')

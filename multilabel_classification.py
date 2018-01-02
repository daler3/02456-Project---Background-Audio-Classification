import matplotlib
matplotlib.use('Agg')
from preprocessor import preprocessor
from keras_models import piczak_CNN_multi
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn import metrics
from keras.models import load_model
import numpy as np
import pandas as pd
import utils
import sklearn.metrics
from sklearn.metrics import f1_score, hamming_loss, zero_one_loss
import matplotlib.pyplot as plt
from pandas_confusion import ConfusionMatrix
import keras.backend as K
import tensorflow as tf

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

def evaluateSavedModel():
	# Choose saved model to evaluate
	model = load_model("saved/Piczak_CNN_pretrainsingle_trainmulti.h5")
	# Setup preprocessor for loading extracted features
	pp = preprocessor(parent_dir='../UrbanSound8K/audio')

	# extracted features that should be loaded to calculate mean and std values
	train_dirs = ["audio_overlap/folder1_overlap", "audio_overlap/folder3_overlap",
				  "audio_overlap/folder4_overlap", "audio_overlap/folder5_overlap", "audio_overlap/folder6_overlap",
				  "audio_overlap/folder7_overlap", "audio_overlap/folder8_overlap", "audio_overlap/folder9_overlap",
				  "audio_overlap/folder10_overlap"]

	#pp.data_prep(train_dirs=[], test_fold="fold2", load_path="../UrbanSound8K/audio/extracted_short_60/")
	# Load features
	test_folder = "fold2"
	pp.load_extracted_fts_lbs(train_dirs=train_dirs, test_fold=test_folder)

	tb = TensorBoard(log_dir='./TensorBoard/piczak_CNN_singlelabel_pretrain_continue_multilabel')
	# model.fit(pp.train_x, pp.train_y,validation_split=.1, epochs=25,
	#		  batch_size=256, verbose=2, callbacks=[tb])



	print("model evaluation")
	scores = model.evaluate(pp.test_x, pp.test_y, verbose=2)
	print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

	# Make predictions
	preds = model.predict(pp.test_x)

	# Evaluate predictions
	evaluateModel(pp, preds, test_folder)


def evaluateModel(pp, preds, fold):
	# ************ PROCESSING THE PREDICTIONS
	preds[preds >= 0.5] = 1
	preds[preds < 0.5] = 0
	print("F1 SCORE:")
	print(f1_score(pp.test_y, preds, average=None))
	print("Hamming Loss:")
	print(hamming_loss(pp.test_y, preds))
	print("Zero-one loss:")
	print(zero_one_loss(pp.test_y, preds))
	# I reach here in plus_one_hot_encode, I want to transform it in one hot
	y_test_preds = utils.from_plus_to_one_hot(np.array(pp.test_y))
	preds_transf = utils.from_plus_to_one_hot(np.array(preds))
	cm = ConfusionMatrix(np.array(y_test_preds).argmax(1), np.array(preds_transf).argmax(1))
	print(cm)
	ax = cm.plot()
	ax.set_xticklabels(classes, rotation="vertical")
	ax.set_yticklabels(classes)
	plt.savefig("cmpre{0}.png".format(fold))

def piczac_cross_validation(epochs, load_path):
	train_dirs = []

	n_folders = 10

	for i in range(1, n_folders + 1):
		#train_dirs.append('fold{0}'.format(i))
		train_dirs.append('folder{0}_overlap'.format(i))

	print(train_dirs)
	for fold in [(9, 10), (2, 5), (6, 7)]:
		val_fold = 'folder{0}_overlap'.format(fold[0])
		test_fold = 'folder{0}_overlap'.format(fold[1])
		#val_fold = 'fold{0}'.format(fold[0])
		#test_fold = 'fold{0}'.format(fold[1])
		train_dirs.remove(val_fold)
		train_dirs.remove(test_fold)

		pp = preprocessor(parent_dir='../../data/UrbanSound8K/audio')
		pp.load_extracted_fts_lbs(train_dirs=train_dirs, val_fold=val_fold, test_fold=test_fold, load_path=load_path)

		model = piczak_CNN_multi(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])
		print("done")
		print("OPTIMIZER")
		#print(model.optimizer.lr)
		#K.set_value(model.optimizer.lr, 0.002)
		#model.optimizer.lr.set_value(0.0001)
		#model.save('Models/model1_all_p2_bn{0}.h5'.format(str(fold)))
		#model = load_model('Models/model1_all_p2{0}.h5'.format(str(fold)))
		#model = load_model('Models/model1_all_p2_bnsec_overlap_{0}.h5'.format(str(fold)))




		tb = TensorBoard(log_dir='./TensorBoard/' + 'overlap_run{0}'.format(fold[1]))
		es = EarlyStopping(patience=10, verbose=1)

		model.fit(pp.train_x, pp.train_y, validation_data=[pp.val_x, pp.val_y], epochs=epochs,
				  batch_size=1000, verbose=2, callbacks=[tb, es])
		#model.save('Models/model1_all_p2_bnsec_overlap_9010_{0}.h5'.format(str(fold)))

		preds = model.predict(pp.test_x)
		evaluateModel(pp,preds, fold)

		K.clear_session()

		train_dirs.append(val_fold)
		train_dirs.append(test_fold)
	# val_fold = 'fold' + str(folds[0])
	# test_fold= 'fold' + str(folds[1])


if __name__ == '__main__':
	K.clear_session()
	tf.reset_default_graph()
	print("main")
	utils.classes_number_mapper()
	print(utils.dic_tot)
	for i in sorted(utils.dic_tot.keys()):
		print(i, utils.dic_tot[i])
	# train_keras_cnn()
	#evaluateSavedModel()
	piczac_cross_validation(10, "../UrbanSound8K/audio/extracted_overlapping_50/audio_overlap")
	#piczac_cross_validation(125, "../../feat_overlap_diff")
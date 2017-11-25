import os
import glob
import numpy as np
import librosa


class preprocessor(object):
	"""
	Preprocessor loads the audio samples, splits them in segments (observations) and saves them along with the
	label in X and y.
	Then, it splits the data in training and test data, saving it in train_x, train_y, test_x and test_y.
	"""
	def __init__(self, parent_dir='data/UrbanSound8K/audio'):
		np.random.seed(23)
		self.parent_dir = parent_dir
		self.train_dirs = ''

		self.X = []
		self.y = []
		self.labels = ([])

		self.train_x = []
		self.train_y = []
		self.test_x = []
		self.test_y = []

	def split_sound_into_segments(self, file_name, segment_size, overlap):
		"""
		Split raw .wav file in librosa segments of segment_size.
		First the method sample_splitter is called to retrieve (start, end) times of the segments.
		Then, the signals are retrieved and collected in the X array along with the labels in another array.

		length of raw sound is 4s or 88200 with sample_rate = 22050Hz
		We have 22050 samples per second, hence 88200 values

		Short variant: default segment_size = hop-length * (frames - 1) = 512 * (41 - 1) = 20480 (why?)
		20480 is ~980ms with sample_rate = 22050Hz

		If 0.5 overlap, segments start at 0, 10240, 20480, ... , 61440, 71680
		88200 / 10240 = 8,61 = 9 segments of length 20480

		Long variant: default segment_size = 512 * (101 - 1) = 51200
		If 0.9 overlap, segments are:
		0 -> 51200,
		(0.1 * 51200) -> 51200 + 0.1 * 51200
		(0.2 * 51200) -> 51200 + 0.2 * 51200
		...
		51200 + 5120 + 5120 + ... Still 9 segments of length 51200

		We conclude that he segments the clips because of the limited amount of training examples.
		Segmenting helps preventing overfitting
		"""

		sound_raw, sample_rate = librosa.load(file_name)

		# normalization
		normalization_factor = 1 / np.max(np.abs(sound_raw))
		sound_raw = sound_raw * normalization_factor

		def sample_splitter(data):
			"""
			This method returns start and end time of the shorter segments produced from the raw sound.
			:param data: raw sound
			:return: (start, end) couples for the new segments
			"""
			start = 0.0
			end = start + segment_size

			while start <= len(data):
				# if last segment
				if (end > len(data)):
					yield int(len(data) - segment_size), int(len(data))
					break

				yield int(start), int(end)
				start += float(segment_size * (1 - overlap))
				end += float(segment_size * (1 - overlap))

		observations = []
		labels = []

		# label is the second part of the filename, i.e. 3 for dog bark
		# TODO: adjust so that it matches how Daniele implements naming of the multiclass files - labels will then just be [1,5] a list of labels for each segment
		label = file_name.split('/')[-1].split('-')[1]

		if len(sound_raw) < segment_size: # one single segment
			sound_raw = np.pad(sound_raw, (0, segment_size - len(sound_raw)), 'constant')
			observations.append(sound_raw)
			labels = np.append(labels, label)
		else:
			for (start, end) in sample_splitter(sound_raw):
				segment = sound_raw[start:end]
				# TODO discard silent segments in a better way?
				if np.any(segment):
					observations.append(segment)
					labels = np.append(labels, label)

		return observations, labels

	def extract_features_cnn(self, sound_raw, segment_size, bands, frames):
		"""
		This method extract features relevant for classification from a librosa representation.
		Bands and frames are the dimensions of the convolution filters applied on the spectrogram.
		"""

		# Since we want 41 frames (might end with 42)
		hop_length = (segment_size / frames)
		hop_length = int(hop_length + 1)  # rounding up

		melspec = librosa.feature.melspectrogram(y=sound_raw, n_mels=bands, hop_length=hop_length)
		logspec = librosa.logamplitude(melspec)
		delta = librosa.feature.delta(logspec)

		features = np.concatenate((logspec.reshape(bands, frames, 1), delta.reshape(bands, frames, 1)), axis=2)
		#features = logspec.reshape(bands, frames, 1)

		return features
	def plus_one_hot_encode(self,labels):
		'''
		Convert labels for single- and multiclass samples into more-than-one-hot encoding.
		'''
		n_labels = len(labels)
		n_unique_labels = 10

		hot_encode = np.zeros((n_labels,n_unique_labels))
		for i,lab in enumerate(labels):
			for j in lab.split('+'):
				hot_encode[i][int(j)] = 1
		return hot_encode


	def one_hot_encode(self, labels):
		"""
		Convert labels to one-hot matrix. Each row is a label, each column is a unique label.
		:param labels:
		:return: one-hot-vector matrix for the labels
		"""
		n_labels = len(labels)
		# n_unique_labels = len(np.unique(labels))
		n_unique_labels = 10
		one_hot_encode = np.zeros((n_labels, n_unique_labels))
		one_hot_encode[np.arange(n_labels), labels] = 1

		return one_hot_encode

	def get_train_test_split(self, test_split=0.2):
		"""
		Generates the split of training/test using the numpy seed
		:param test_split:
		:return:
		"""
		n_samples = len(self.X)

		# Create a permutation
		random_perm = np.random.permutation(n_samples)
		# Train length
		n_train = int(np.round(n_samples * (1. - test_split)))

		test_x = np.array([self.X[s] for s in random_perm[n_train:]])
		test_y = np.array([self.y[s] for s in random_perm[n_train:]])
		train_x = np.array([self.X[s] for s in random_perm[:n_train]])
		train_y = np.array([self.y[s] for s in random_perm[:n_train]])

		return train_x, train_y, test_x, test_y

	def data_prep(self, train_dirs, test_fold='', segment_size=20480, overlap=0.5, bands=60, frames=41, file_ext="*.wav",
				  save_path='', load_path=''):
		"""
		Data prep loads all the sound files in train_dirs, then it splits them into segments of segment_size,
		then it extracts features and labels. Finally, it splits the data in train and test and assigns these to variables.
		"""

		self.train_dirs = train_dirs

		if load_path:
			for dir in train_dirs:
				fts = np.load(load_path + '/' + dir + '/features.npy')
				lbs = np.load(load_path + '/' + dir + '/labels.npy')
				if len(self.X) == 0:
					self.X, self.y = fts, lbs
				else:
					self.X = np.concatenate((self.X, fts), axis=0)
					self.y = np.concatenate((self.y, lbs), axis=0)
		else:
			X_total, labels_total = [], []

			for sub_dir in self.train_dirs:
				for fn in glob.glob(os.path.join(self.parent_dir, sub_dir, file_ext)):
					try:
						# data
						segments, labels = self.split_sound_into_segments(fn, segment_size, overlap)
						# features
						[X_total.append(self.extract_features_cnn(sample, segment_size, bands, frames)) for sample in segments]
						# labels
						[labels_total.append(label) for label in labels]
						print(fn)
					except Exception as e:
						print ("Error encountered while parsing file: ", fn, e)
						continue
				if save_path:
					directory = save_path + '\\' + sub_dir
					if not os.path.exists(directory):
						os.makedirs(directory)
					np.save(directory + '\\features', np.array(X_total))
					np.save(directory + '\\labels', self.one_hot_encode(np.array(labels_total, dtype=np.int)))
					X_total, labels_total = [], []

			self.labels = labels_total
			self.y = self.plus_one_hot_encode(labels_total)
			#self.y = self.one_hot_encode(np.array(labels_total, dtype=np.int))
			self.X = np.array(X_total)

		if test_fold:
			self.train_x, self.train_y = self.X, self.y
			self.test_x = np.load(load_path + '/' + test_fold + '/features.npy')
			self.test_y = np.load(load_path + '/' + test_fold + '/labels.npy')
		else:
			self.train_x, self.train_y, self.test_x, self.test_y = self.get_train_test_split()


if __name__ == '__main__':
	# Testing the data_preprocessor
	pp = preprocessor()

	# Run this to extract features and save them
	# (First you have to create the folders)
	# pp.data_prep(train_dirs=["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"], save_path='extracted')

	print("Breakpoint here!")

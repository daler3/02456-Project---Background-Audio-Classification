import numpy as np
import glob
import os
import librosa


class preprocessor(object):

    def __init__(self, parent_dir='data/UrbanSound8K/audio'):
        np.random.seed(23)
        self.parent_dir = parent_dir
        self.X = []
        self.y = []
        self.train = ([],[])
        self.test = ([],[])
        self.labels = ([])

    def split_sound_into_segments(self, file_name, segment_size=20480):
        """
        Split raw .wav file in librosa segments of segment_size.
        First the method sample_splitter is called to retrieve (start, end) times of the segments.
        Then, the signals are retrieved and collected in the X array along with the labels in another array.
        :param file_name:
        :param segment_size:
        :return:

        length of raw sound is 4s or 88200 with sample_rate = 22050Hz
        default segment_size = 512 * (frames - 1) = 512 * 40 = 20480 (why?)
        20480 is ~980ms with sample_rate = 22050Hz

        If 0.5 overlap, segments start at 0, 10240, 20480, etc.
        88200 / 10240 = 8,61 = 9 segments of length 20480
        """

        sound_raw, sample_rate = librosa.load(file_name)

        # TODO Normalize sound
        # Accrding to its own max?
        #normalization_factor = 1 / np.max(np.abs(sound_raw))
        #sound_raw = sound_raw * normalization_factor

        X = []
        labels = []

        def sample_splitter(data, overlap=0.5):
            """
            This method returns start and end time of the shorter segments produced from the raw sound.
            :param data: raw sound
            :param overlap: overlap between one segment and the other
            :return: (start, end) couples for the new segments
            """
            start = 0.0
            while (start + segment_size) < len(data):
                yield int(start), int(start + segment_size)
                # 50% overlap
                start += float(segment_size * overlap)

        # label is the first part of the filename, i.e. 7383 for dog bark
        label = file_name.split('/')[-1].split('-')[0]

        for (start, end) in sample_splitter(sound_raw):
            signal = sound_raw[start:end]
            X.append(signal)
            labels = np.append(labels, label)

        return X, labels

    def extract_features_cnn(self, sound_raw, segment_size, bands, frames):
        """
        This method extract features relevant for classification from a librosa representation.
        Bands and frames are the dimensions of the convolution filters applied on the spectrogram.
        :param sound_raw:
        :param segment_size:
        :param bands:
        :param frames:
        :return:
        """

        # Since we want 41 frames (might end with 42)
        hop_length = (segment_size / frames)
        hop_length = int(hop_length + 1)  # dummy + 1 for always rounding up TODO why?

        melspec = librosa.feature.melspectrogram(sound_raw, n_mels=bands, hop_length=hop_length)
        logspec = librosa.logamplitude(melspec)
        delta = librosa.feature.delta(logspec)

        features = np.concatenate((logspec.reshape(bands, frames, 1), delta.reshape(bands, frames, 1)), axis=2)

        return features

    def one_hot_encode(self, labels):
        """
        Convert labels to one-hot matrix. Each row is a label, each column is a unique label.
        :param labels:
        :return: one-hot-vector matrix for the labels
        """
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1

        return one_hot_encode

    def get_train_test_split(self, test_split=0.2):
        """
        Generates the split of training/test using the numpy seed
        :param test_split:
        :return:
        """
        print ("Creating data split split...")
        n_samples = len(self.X)

        # Create a permutation
        random_perm = np.random.permutation(n_samples)
        # Train length
        n_train = int(np.round(n_samples * (1. - test_split)))

        test_x = np.array([self.X[s] for s in random_perm[n_train:]])
        test_y = np.array([self.y[s] for s in random_perm[n_train:]])
        train_x = np.array([self.X[s] for s in random_perm[:n_train]])
        train_y = np.array([self.y[s] for s in random_perm[:n_train]])

        print ("Length of trainX: {0}".format(len(train_x)))
        print ("Length of trainY: {0}".format(len(train_y)))
        print ("Length of TestX: {0}".format(len(test_y)))
        print ("Length of TestY: {0}".format(len(test_y)))

        train = (train_x, train_y)
        test = (test_x, test_y)

        return train, test

    def data_prep(self, sub_dirs, segment_size=2048, bands=60, frames=41, file_ext="*.wav"):
        """
        Data prep loads all the sound files in sub_dirs, then it splits them into segments of segment_size,
        then it extracts features and labels. Finally, it splits the data in train and test and assigns these to variables.
        :param sub_dirs:
        :param frames:
        :param segment_size:
        :param bands:
        :param file_ext:
        :return:
        """

        X_total, labels_total = [], []

        for sub_dir in sub_dirs:
            for fn in glob.glob(os.path.join(self.parent_dir, sub_dir, file_ext)):
                try:
                    X, labels = self.split_sound_into_segments(fn, segment_size)
                    [X_total.append(self.extract_features_cnn(sample, segment_size, bands, frames)) for sample in X]
                    [labels_total.append(label) for label in labels]
                except Exception as e:
                    print ("Error encountered while parsing file: ", fn)
                    continue

        self.labels = labels_total
        self.y = self.one_hot_encode(np.array(labels_total, dtype=np.int))
        self.X = np.array(X_total)
        self.train, self.test = self.get_train_test_split()

if __name__ == '__main__':
    # Testing the data_preprocessor
    pp = preprocessor()

    pp.data_prep(sub_dirs=["fold1"])
    print("Breakpoint here!")

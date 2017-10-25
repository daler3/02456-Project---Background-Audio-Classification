"""
Initial analysis. Load .wav sound files and plot waves, specgram and log_power_specgram.
Credits to http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

Change filepaths, sound names and figsize for matplotlib
"""
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def plot_waves(sound_names,raw_sounds):
    i = 1
    plt.figure(figsize=(12,6))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(4,2,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()


def plot_specgram(sound_names,raw_sounds):
    i = 1
    plt.figure(figsize=(12,6))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(4,2,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    plt.figure(figsize=(12,6))
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(4, 2, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram", x=0.5, y=0.915, fontsize=18)
    plt.show()


if __name__ == '__main__':
    sound_file_paths = ["data/UrbanSound8K/audio/fold1/7383-3-0-0.wav"]

    sound_names = ["dog bark"]

    raw_sounds = load_sound_files(sound_file_paths)

    plot_waves(sound_names,raw_sounds)
    plot_specgram(sound_names,raw_sounds)
    plot_log_power_specgram(sound_names,raw_sounds)

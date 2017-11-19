import pandas
import sys
import wave
import math
import struct
import random
import argparse
from itertools import *
import pydub 
from pydub import AudioSegment
from pydub import generators as g


#def white_noise(amplitude=0.5):
#    return (float(amplitude) * random.uniform(-1, 1) for _ in count(0))

def add_white_noise(sound1, duration=1700.0, volume=-15.0):
    wn = g.WhiteNoise().to_audio_segment(duration=duration, volume=volume)
    combined = sound1.overlay(wn)
    return combined 

def combine_sounds(sound1, sound2, output_path):
    combined = sound1.overlay(sound2)
    combined.export(output_path, format='wav')

def retrieve_audio_from_df(or_path, element_df):
    rs = element_df['slice_file_name'].to_string(header=False, index=False).split(";")
    if len(rs) > 1: ##then wn
        name = rs[0]
        path = or_path + "/" + name
        sound = AudioSegment.from_file(path)
        sound_with_wn = add_white_noise(sound)
    else: 
        path = or_path + "/" + rs
        sound = AudioSegment.from_file(path)
    return sound


df = pandas.read_csv("UrbanSound8K.csv")
df_fold1 = df.loc[df['fold'] == 1]

target_number = 100

###get all classes separately 
air_conditioner_list = df_fold1.loc[df_fold1['class'] == 'air_conditioner']
car_horn_list = df_fold1.loc[df_fold1['class'] == 'car_horn']
children_playing_list = df_fold1.loc[df_fold1['class'] == 'children_playing']
dog_bark_list = df_fold1.loc[df_fold1['class'] == 'dog_bark']
drilling_list = df_fold1.loc[df_fold1['class'] == 'drilling']
engine_idling_list = df_fold1.loc[df_fold1['class'] == 'engine_idling']
gun_shot_list = df_fold1.loc[df_fold1['class'] == 'gun_shot']
jackhammer_list = df_fold1.loc[df_fold1['class'] == 'jackhammer']
siren_list = df_fold1.loc[df_fold1['class'] == 'siren']
street_music_list = df_fold1.loc[df_fold1['class'] == 'street_music']

class_list = [air_conditioner_list, car_horn_list, children_playing_list, 
             dog_bark_list, drilling_list, engine_idling_list, gun_shot_list,
             jackhammer_list, siren_list, street_music_list]

new_lists = []
for c in class_list: 
    if len(c) >= target_number: 
        new_lists.append(c)
        continue
    else: 
        el_needed = target_number - len(c)
        index = 0
        for i in range (0, el_needed):
            element = c.iloc[[index]]
            rs = element['slice_file_name'].to_string(header=False, index=False).split(";")
            if len(rs) == 1:
                element['slice_file_name'] = element['slice_file_name']+";wn"
            ###put something in the name of the element
            ###that makes me understanding that I have to add white noise
            c = c.append(element, ignore_index=True)  
            index = index + 1
    new_lists.append(c)
    
index = 0
for l in new_lists: 
    new_lists[index] = l.reset_index(drop=True)
    index += 1

#shuffle
index = 0
for l in new_lists: 
    new_lists[index] = new_lists[index].sample(frac=1).reset_index(drop=True)
    index += 1


##take one by one and combine##
copy_new_lists = new_lists.copy()
destination_path = "data/UrbanSound8K/audio_overlap/folder1_overlap"
original_path = "data/UrbanSound8K/audio/folder1"
new_csv_list = pandas.DataFrame
for l in copy_new_lists: 
    for el in l: 
        #retrieve name and row df
        #from path load
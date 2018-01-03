
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
import scipy.io
from scipy.io import wavfile
import soundfile as sf
import wavio



def add_white_noise(sound1, duration=1700.0, volume=-17.0):
    wn = g.WhiteNoise().to_audio_segment(duration=duration, volume=volume)
    combined = sound1.overlay(wn)
    return combined 

def combine_sounds(sound1, sound2, output_path, el_name):
    combined = sound1.overlay(sound2)
    output_path = output_path + "/" + el_name
    combined.export(output_path, format='wav')

def retrieve_audio_from_df(or_path, element_df):
    rs = element_df['slice_file_name'].to_string(header=False, index=False).split(";")
    if len(rs) > 1: ##then wn
        name = rs[0]
        path = or_path + "/" + name
        sound = AudioSegment.from_file(path)
        sound_with_wn = add_white_noise(sound)
        sound = sound_with_wn
    else: 
        path = or_path + "/" + rs[0]
        sound = AudioSegment.from_file(path)

    return sound

def combine_elements_names(el1, el2, folder_n):
    s_file_name1 = (el1['slice_file_name'].to_string(header=False, index=False).split(";"))[0]
    s_file_name2 = (el2['slice_file_name'].to_string(header=False, index=False).split(";"))[0]
    splitted_file_name1 = s_file_name1.split(".")[0].split("-")
    splitted_file_name2 = s_file_name2.split(".")[0].split("-")
    res_file_name = splitted_file_name1[0]+"+"+splitted_file_name2[0] + "-" + splitted_file_name1[1]+"+"+splitted_file_name2[1] + "-" + splitted_file_name1[2]+"+"+splitted_file_name2[2] + "-" + splitted_file_name1[3]+"+"+splitted_file_name2[3] + ".wav"
    
    fsID1 = el1['fsID'].to_string(header=False, index=False)
    fsID2 = el2['fsID'].to_string(header=False, index=False)
    res_fsID = fsID1+"+"+fsID2
        
    start1 = el1['start'].to_string(header=False, index=False)
    start2 = el2['start'].to_string(header=False, index=False)
    res_start = start1+"+"+start2
    
    end1 = el1['end'].to_string(header=False, index=False)
    end2 = el2['end'].to_string(header=False, index=False)
    res_end = end1+"+"+end2
    
    salience1 = el1['salience'].to_string(header=False, index=False)
    salience2 = el2['salience'].to_string(header=False, index=False)
    res_salience = salience1+"+"+salience2
    
    ###folder
    folder_name = str(folder_n)+"_combined" 
    
    classID1 = el1['classID'].to_string(header=False, index=False)
    classID2 = el2['classID'].to_string(header=False, index=False)
    res_classID = classID1+"+"+classID2
    
    class1 = el1['class'].to_string(header=False, index=False)
    class2 = el2['class'].to_string(header=False, index=False)
    res_class = class1+"+"+class2
    
    return pandas.DataFrame([[res_file_name, res_fsID, res_start, res_end, res_salience, folder_name, res_classID, res_class]], columns=list(el1.columns.values)), res_file_name


def overlay_50_50_creation():
    for nf in range (1, 11):
        #fold_n = 2
        fold_n = nf
        df = pandas.read_csv("data/UrbanSound8K/metadata/UrbanSound8K.csv")
        df_fold1 = df.loc[df['fold'] == fold_n]

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


        folder_number = str(fold_n)
        ##take one by one and combine##
        copy_new_lists = new_lists.copy()
        destination_path = "data/UrbanSound8K/audio_overlap/folder"+folder_number+"_overlap"
        original_path = "data/UrbanSound8K/audio/fold"+folder_number

        maxim = 9
        range_ind = 99
        for i in range(0, len(new_lists)-1): ###list of the lists
        	l = copy_new_lists[0]
        	copy_new_lists.remove(l) #remove that list
        	index = 0
        	for counter_ind in range(0, range_ind): ##for each element in the first class
        		el = l.iloc[[counter_ind]]
        		if index == maxim: 
        			index = 0
        		#retrieve name and row df
        		#from path load
        		sound1 = retrieve_audio_from_df(original_path, el) ##retrieve original sound
        		sound2 = retrieve_audio_from_df(original_path, copy_new_lists[index].iloc[[0]]) #retrieve second sound
        		###combine the two sounds and save the results
        		res_element_row, el_name = combine_elements_names(el, copy_new_lists[index].iloc[[0]], folder_number) #name of the combined sound
        		combine_sounds(sound1, sound2, destination_path, el_name)
        		if (i==0 and counter_ind == 0):
        			new_csv_list = res_element_row

        		else:
        			new_csv_list = new_csv_list.append(res_element_row, ignore_index=True) ##append the element to the pandas df, to be written

        		copy_new_lists[index].drop(0, axis=0, inplace=True)
        		copy_new_lists[index].reset_index(drop=True, inplace=True) ##remove that element
        		index += 1 ## increase  
        	maxim = maxim - 1
        	range_ind = range_ind - 11

        ##save the new cs to a file 
        new_csv_list.to_csv("data/UrbanSound8K/audio_overlap/folder_"+folder_number+"_names.csv")
# Does the file exist?
# Is the size right?
# 

import csv, sys
import os
import dl_utils

filename = 'balanced_train_segments.csv'
rownum = 0
path = 'E:/AudioSet/balanced_train/'
testpath = 'E:/AudioSet/test/'
testVid = '-0DdlOuIFUI.wav'
logfile_path = 'E:/AudioSet/logfile.txt'\

def download(row):
    url = row[0]
    start_time = float(row[1].lstrip())
    end_time = float(row[2].lstrip())
    ydl_opts = {}
    video = dict
    try:
        video_duration = dl_utils.get_video_duration(url)
    except Exception as error:
        with open(logfile_path, 'a', encoding='utf8') as logfile:
            print('Download Failed at row: ' + str(rownum) + ' with error: ' + '{}'.format(error), file=logfile)              
    if video_duration - start_time < 10:
        duration = video_duration - (start_time + 1) #Download one tenth of a second less as the duration is rounded up.
    else:
        duration = end_time - (start_time + 1)
    ret = dl_utils.youtube_downloader(url, str(start_time), str(duration), path)
    if ret != 0:
        dl_utils.create_error_file(url, str(rownum), path)  

# dl_utils.file_exists(testVid, path)
# dl_utils.filenames_remove_index(path)
fileNames = {}
realFileNames = {}
for entry in os.scandir(path):
    fileNames[entry.name] = entry
with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    try:
        for row in data:
            if rownum >= 3:
                url = row[0]
                vIdFilename = url + '.wav'
                realFileNames[vIdFilename] = row

                # if vIdFilename not in fileNames:
                #     #Try Download again
                #     download(row)                
            rownum += 1
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

# Compare filenames with realfilenames and remove ones that should not be there
# wrong_filenames = fileNames.keys() - realFileNames.keys()
# for wrong_file in wrong_filenames:
#     dl_utils.delete_file(wrong_file, path)

# Check if length of downloaded files are correct
for file_name in fileNames:
    try:
        file_length = dl_utils.get_wav_file_length(path, file_name)
        row = realFileNames[file_name]
        start_time = float(row[1].lstrip())
        end_time = float(row[2].lstrip())
        real_duration = end_time - start_time
        duration_diff = real_duration - file_length
        if abs(duration_diff) > 1.2:
            # Delete file and download again
            dl_utils.delete_file(file_name, path)
            print('duration should be ' + str(real_duration))
            print('duration is ' + str(file_length))
            print(row)
            download(row)
    except Exception as error:
        print(error)
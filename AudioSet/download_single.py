import csv, sys
import os
import subprocess
import youtube_dl

filename = 'balanced_train_segments.csv'
path = 'E:/AudioSet/balanced_train/'

row_to_download = 13225


def youtube_download_os_call(vId, start_time, duration, idx):    
    command = 'ffmpeg -n -ss ' + start_time + ' -t ' + duration +' -i $(youtube-dl -i -w --extract-audio --audio-format wav --audio-quality 0 --get-url https://www.youtube.com/watch?v=' + vId + ') -t ' + duration + ' ' + path + idx + '_' + vId + '.wav'
    # This is used in windows as the PATH is not taken into consideration in os.system
    
    try:
        ret = subprocess.run(['powershell','-command', command], timeout=10).returncode
    except:
        ret = -1
        # ret.returncode = -1
    
    # This will propably work on Linux and macOS
    # ret = os.system('ffmpeg -n -ss ' + start_time +
    #           ' -i $(youtube-dl -i -w --extract-audio '
    #           '--audio-format wav --audio-quality 0 '
    #           '--get-url https://www.youtube.com/watch?v=' + id + ')'
    #           ' -t 10 ' + path + idx + '_' + id + '.wav')

    return ret

def create_error_file(vId, idx):
    with open(path + idx + '_' + vId + '_ERROR.wav', 'a'):
        os.utime(path + idx + '_' + vId + '_ERROR.wav', None)

def youtube_downloader(vId, start_time, duration, idx):    
    print('ffmpeg -n -ss ' + start_time + ' -t ' + duration +' -i $(youtube-dl -i -w --extract-audio --audio-format wav --audio-quality 0 --get-url https://www.youtube.com/watch?v=' + vId + ') -t ' + duration + ' ' + path + idx + '_' + vId + '.wav')
    ret = youtube_download_os_call(vId, start_time, duration, idx)
    return ret

with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    row_count = len(data)
    row = data[row_to_download + 3]
    try:
        start_time = float(row[1].lstrip())
        end_time = float(row[2].lstrip())
        ydl_opts = {}
        video = dict
        url = row[0]
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                video = ydl.extract_info('https://www.youtube.com/watch?v='+url, download=False)
            except:
                pass                   
        video_duration = video['duration']
        if video_duration - start_time < 10:
            duration = video_duration - (start_time + 0.1) #Download one tenth of a second less as the duration is rounded up.
        else:
            duration = end_time - (start_time + 0.1)
        ret = youtube_downloader(url, str(start_time),str(duration), str(row_to_download))
        # If there was an error downloading the file
        # This sometimes happens if videos are blocked or taken down
        if ret != 0:
            create_error_file(url, str(row_to_download))  
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
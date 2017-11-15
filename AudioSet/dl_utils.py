import subprocess
import os
import re
import youtube_dl
import wave
import contextlib

def youtube_download_os_call(vId, start_time, duration, path) :    
    command = 'ffmpeg -n -ss ' + start_time + ' -t ' + duration +' -i $(youtube-dl -i -w --extract-audio --audio-format wav --audio-quality 0 --get-url https://www.youtube.com/watch?v=' + vId + ') -t ' + duration + ' ' + path + vId + '.wav'
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

def youtube_downloader(vId, start_time, duration, path):    
    print('ffmpeg -n -ss ' + start_time + ' -t ' + duration +' -i $(youtube-dl -i -w --extract-audio --audio-format wav --audio-quality 0 --get-url https://www.youtube.com/watch?v=' + vId + ') -t ' + duration + ' ' + path + vId + '.wav')
    ret = youtube_download_os_call(vId, start_time, duration, path)
    return ret

def create_error_file(vId, idx, path):
    with open(path + idx + '_' + vId + '_ERROR.wav', 'a'):
        os.utime(path + idx + '_' + vId + '_ERROR.wav', None)

def file_exists(vId, path):
    for entry in os.scandir(path):
        if vId in entry.name:
            print(entry)

def filenames_remove_index(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            newFileName = re.sub('^[^_]*_', '', f) # Regex to find everything before _
            os.replace(os.path.join(path,f), os.path.join(path,newFileName)) # Apparently this does not work as intended as the updated files also is searched again thereby removing some of the actual filename

# Gets total duration of youtube video by vId/url
def get_video_duration(url):
    ydl_opts = {'quiet': True}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            video = ydl.extract_info('https://www.youtube.com/watch?v='+url, download=False)
            return video['duration']
        except Exception as error:         
            raise error
# Deletes a file
def delete_file(file_name, path):
    os.remove(os.path.join(path,file_name))

# Returns the length of a wave file in seconds
def get_wav_file_length(path, file_name):
    sample = path + file_name
    with contextlib.closing(wave.open(sample, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
        # print(length)
    return length
import csv, sys
import os
import contextlib
import wave
import subprocess
import youtube_dl
from dl_utils import youtube_downloader, create_error_file

filename = 'balanced_train_segments.csv'
rownum = 0
path = 'E:/AudioSet/balanced_train/'

# specify the index of files that is downloaded last time (to resume downloading)
# Basically this is a simple work around for this downloader, where I sometimes accidentally close the program or sometimes it just hangs in my environment
last_processed_row = 0

with open(filename, newline='') as f:
    reader = csv.reader(f)
    try:
        for row in reader:
            # Start with row just after last processed
            if rownum <= last_processed_row + 3:
              rownum += 1
              continue
            # Skip the 3 line header
            if rownum >= 3:
                # print(row)
                start_time = float(row[1].lstrip())
                end_time = float(row[2].lstrip())
                ydl_opts = {'quiet': True}
                video = dict
                url = row[0]
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    try:
                       video = ydl.extract_info('https://www.youtube.com/watch?v='+url, download=False)
                    except:
                        rownum += 1
                        continue                    
                video_duration = video['duration']
                if video_duration - start_time < 10:
                    duration = video_duration - start_time
                else:
                    duration = end_time - start_time
                ret = youtube_downloader(url, str(start_time),str(duration), path)
                # If there was an error downloading the file
                # This sometimes happens if videos are blocked or taken down
                if ret != 0:
                    create_error_file(url, str(rownum - 3), path)

            rownum += 1
            if(rownum % 1000 == 0): 
                print(rownum)
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
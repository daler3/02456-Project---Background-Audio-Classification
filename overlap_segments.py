# import wave

# infiles = ["prova_stack/7061-6-0-0.wav", "prova_stack/7383-3-0-0.wav"]
# outfile = "prova_stack/sounds.wav"

# data= []
# for infile in infiles:
#     w = wave.open(infile, 'rb')
#     data.append( [w.getparams(), w.readframes(w.getnframes())] )
#     w.close()

# output = wave.open(outfile, 'wb')
# output.setparams(data[0][0])
# output.writeframes(data[0][1])
# output.writeframes(data[1][1])
# output.close()
import sys
import wave
import math
import struct
import random
import argparse
from itertools import *


def white_noise(amplitude=0.5):
    return (float(amplitude) * random.uniform(-1, 1) for _ in count(0))

import pydub 
from pydub import AudioSegment
from pydub import generators as g

sound1 = AudioSegment.from_file("prova_stack/7061-6-0-0.wav")
sound2 = AudioSegment.from_file("prova_stack/7383-3-0-0.wav")
wn = g.WhiteNoise().to_audio_segment(duration=1700.0, volume=-15.0)
combined = sound1.overlay(wn)

combined.export("prova_stack/sounds.wav", format='wav')
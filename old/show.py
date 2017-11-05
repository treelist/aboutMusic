
import numpy as np

import scipy
import scipy.io.wavfile
from scipy import fftpack
import matplotlib.pyplot as plt

WAVDIR1 = "./band/"
WAVDIR2 = "./canon/"
LISTNAME1 = "list_test.txt"
LISTNAME2 = "list02.txt"

with open(LISTNAME1, "r") as f:
	wavs1 = f.readlines()
with open(LISTNAME2, "r") as f:
	wavs2 = f.readlines()

	
wav_filename = "001_G.wav"

plt.clf()
sample_rate, X = scipy.io.wavfile.read(WAVDIR1 + wav_filename)
spectrum = fftpack.fft(X)
freq = fftpack.fftfreq(len(X), d = 1.0 / sample_rate)
print("freq:", freq, "freq.shape", freq.shape)

num_sample = len(X)
plt.xlabel("time [s]")
plt.title(wav_filename)
plt.plot(np.arange(num_sample) / sample_rate, X[:int(num_sample)])
plt.grid(True)

plt.show()

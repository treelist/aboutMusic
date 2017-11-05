
import numpy as np

import scipy
import scipy.io.wavfile
from scipy import fftpack
import matplotlib.pyplot as plt

WAVDIR1 = "./band/"
WAVDIR2 = "./canon/"
LISTNAME1 = "list_test.txt"
LISTNAME1 = "list01.txt"
LISTNAME2 = "list02.txt"

with open(LISTNAME1, "r") as f:
	wavs1 = f.readlines()
with open(LISTNAME2, "r") as f:
	wavs2 = f.readlines()

hzList = [
	[16.4, 17.3, 18.4, 19.4, 20.6, 21.8, 23.1, 24.5, 26.0, 27.5, 29.1, 30.9],
	[32.7, 34.6, 36.7, 38.9, 41.2, 43.7, 46.2, 49.0, 51.9, 55.0, 58.3, 61.7],
	[65.4, 69.3, 73.4, 77.8, 82.4, 87.3, 92.5, 98.0, 103.8, 110.0, 116.5, 123.5],
	[130.8, 138.6, 146.8, 155.6, 164.8, 174.6, 185.0, 196.0, 207.7, 220.0, 233.1, 246.9],
	[261.6, 277.2, 293.7, 311.1, 329.6, 349.2, 370.0, 392.0, 415.3, 440.0, 466.2, 493.9],
	[523.3, 554.4, 587.3, 622.3, 659.3, 698.5, 740.0, 784.0, 830.6, 880.0, 932.3, 987.8],
	[1046.5, 1108.7, 1174.7, 1244.5, 1318.5, 1396.9, 1480.0, 1568.0, 1661.2, 1760.0, 1864.7, 1975.5],
	[2093.0, 2217.5, 2349.3, 2489.0, 2637.0, 2793.8, 2960.0, 3136.0, 3322.4, 3520.0, 3729.3, 3951.1],
]
hzList_flatten =  [item for sublist in hzList for item  in sublist]
xticks = []
for i in range(len(hzList_flatten)):
	if i % 12 == 11:
		xticks.append(str(hzList_flatten[i]))
	else:
		xticks.append(" ")
	
a = 2 ** (1.0 / 24.0)
hzBorderList = [i / a for i in hzList_flatten]
hzBorderList.append(hzList_flatten[-1] * a)

CUTOFF = 0.5
def tryStacking(freq, spectrum, sample_rate):
	if(len(freq) != len(spectrum)):
		print("Error in 'if(len(freq) != len(spectrum))!'")
		return -1
	
	fr = freq[:int(len(freq) / 2)]
	sp = abs(spectrum[:int(len(spectrum) / 2)]) / sample_rate
	sp[sp < CUTOFF] = 0

	step = [] v 
	for i in range(len(hzBorderList)):
		step.append(np.argmax(fr > hzBorderList[i]))
	
	splist = np.split(sp, step)
	del splist[0]
	del splist[-1]
	if(len(splist) != len(hzList_flatten)):
		print("Error in 'if(len(splist) == len(hzList_flatten))'!")
		print("len(splist):", len(splist), ", len(hzList_flatten):", len(hzList_flatten))
	
	sum = [arr.sum() for arr in splist]
	sumArr = np.array(sum)
	sumArr = sumArr / sumArr.sum()
	print(sumArr.sum())
	return sumArr

def makedat(wavlist, wavpath):
	for i in wavlist:
		plt.clf()
		print(wavpath + i.strip())
		sample_rate, X = scipy.io.wavfile.read(wavpath + i.strip())
		spectrum = fftpack.fft(X)
		freq = fftpack.fftfreq(len(X), d = 1.0 / sample_rate)
		#print("Sample_rate:", sample_rate)
		#print("freq:", freq, "freq.shape", freq.shape)

		#plt.xlim(0, 4000)
		#plt.xlim(xmin=0)
		
		#plt.xlabel("frequency [Hz]")
		#plt.xticks(np.arange(10) * 400)
		#plt.xticks(np.array(hzList_flatten))

		fft_desc = i.strip()
		#plt.title("FFT of %s" % fft_desc)
		plt.title("ratio of pitches of %s" % fft_desc)
		#plt.plot(freq[:len(freq)/2], abs(spectrum[:len(freq)/2]) / sample_rate, linewidth = 2)
		#plt.plot(freq, abs(spectrum) / sample_rate,  linewidth = 2)
		#plt.grid(True)
		plt.tight_layout()
		
		sum = tryStacking(freq, spectrum, sample_rate)
		plt.xticks(range(len(sum)), xticks)
		plt.xlim(12, len(sum))
		plt.bar(range(len(sum)), sum, width = 1.0)
		
		np.save(fft_desc, sum)
		np.savetxt(fft_desc + ".txt", sum)
		
		plt.savefig("%s_wav.png" % i.split(".")[0], bbox_inches = 'tight')


makedat(wavs1, WAVDIR1)
makedat(wavs2, WAVDIR2)


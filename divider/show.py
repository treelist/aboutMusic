
import numpy as np

from scipy import fftpack
import matplotlib.pyplot as plt

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

# hzList의 값을 한 리스트에 넣기 (hzList의 값 sublist에 대해), (sublist의 값 item)에 대해 컴프리헨션
hzList_flatten =  [item for sublist in hzList for item in sublist]

def show_amp(filename, data, sample_rate):
    plt.clf() # clear figure, 그냥
    plt.xlabel("time [s]") # x축의 단위는 초
    plt.title(filename) # 표 이름
    
    plot_x = np.arange(len(data)) / sample_rate # (1 / sample_rate)초 단위
    plot_y = data
    plt.plot(plot_x, plot_y)
    
    plt.grid(True) # 보기 좋음
    plt.show()

def show_freq(filename, data, sample_rate):
    plt.clf()
    plt.xlabel("frequency [Hz]") # 단위는 주파수
    plt.title("ratio of pitches")
    
    spectrum = fftpack.fft(data) # fast fourier transform
    freq = fftpack.fftfreq(len(data), d = 1.0 / sample_rate) # spectrum에 맞춰줄 값
    
    plt.xlim(0, 4000) # 4000Hz 이하만 다룹
    plt.xlim(xmin=0)
    
    plot_x = freq[:len(freq) // 2] # 대칭임, 앞 부분이 0보다 큰 값을 가지므로 반만 사용
    plot_y = abs(spectrum[:len(freq) // 2]) / sample_rate # spectrum은 complex number. abs로 값을 구함. 역시 대칭이므로 반만. 너무 큰 값이므로 나눔.
    plt.plot(plot_x, plot_y, linewidth = 2)
    
    plt.grid(True) # 보기 좋음
    plt.show()

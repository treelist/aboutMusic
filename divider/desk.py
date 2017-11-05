
import numpy as np
import scipy
import scipy.io.wavfile
import show
import find_min_interval
import matplotlib.pyplot as plt

DIRPATH = "./sample/" # wav 파일이 있는 위치
filename = "canon_mono.wav" # wav 파일 이름
#filename = "001_G.wav"


# wav 파일을 받아서 샘플링 레이트, 변위 데이터 구하기
# data 는 (1 / sample_rate)초 단위로 구한 변위
sample_rate, data = scipy.io.wavfile.read(DIRPATH + filename)



#show.show_freq(filename, data, sample_rate)
#show.show_amp(filename, data, sample_rate)

rate = sample_rate // 60
lst = []

for i in range(len(data) // 60):
    lst.append(np.abs(data[i * 60 : (i + 1) * 60]).mean())

rate2 = sample_rate // 10
lst2 = []

for i in range(len(data) // 10):
    lst2.append(np.abs(data[i * 10 : (i + 1) * 10]).mean())

rate3 = sample_rate // 735
lst3 = []

for i in range(len(data) // 735):
    lst3.append(np.abs(data[i * 735 : (i + 1) * 735]).mean())


print("sample_rate:", sample_rate)
print("len(data):", len(data))

print("rate:", rate)
print("len(lst):", len(lst))

#show.show_amp(filename, data, sample_rate)
show.show_amp(filename, np.abs(data), sample_rate)
show.show_amp(filename, lst, rate)
show.show_amp(filename, lst3, rate3)
#show.show_amp(filename, lst2, rate)

rr = find_min_interval.find_best(np.array(lst3), 30, 1000)
plt.plot(rr[:,0], rr[:,1])
plt.show()



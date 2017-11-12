
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

import toolbox
import paintbox
import setting

DIRPATH = "./sample/" # wav 파일이 있는 위치
filename = "canon_mono.wav" # wav 파일 이름

# wav 파일을 읽습니다. sample_rate에는 wav의 샘플레이트, data에는 1 / (샘플레이트) 초 단위의 변위(진폭)값이 할당됩니다.
sample_rate, data = scipy.io.wavfile.read(DIRPATH + filename)
data = data.astype("int32") # 오버플로우 방지

MY_SAMPLE_RATE = 30                    # 이 example에서 단순화 해서 사용할 sample_rate입니다.
value =  sample_rate // MY_SAMPLE_RATE # 원래 샘플레이트와의 비율입니다.

# 샘플레이트 30으로 음의 세기를 나타낼 ndarray입니다.
simplified_abs_data = toolbox.simplifier(data, value, abs=True)

print("WAV sample rate:", sample_rate)
print("len(data):", len(data))
print("====================")
print("my sample rate:", MY_SAMPLE_RATE)
print("RATIO(sample rate / my sample rate):", value)
print("len(simplified_abs_data):", len(simplified_abs_data))

# 저장용 이름을 설정합니다. 파일명_{MY_SAMPLE_RATE}
title_without_ext = ".".join(filename.split(".")[:-1])
title = title_without_ext + "_" + str(MY_SAMPLE_RATE)

# 처음 얻은 데이터로 곡 전체를 보여줍니다.
paintbox.show_amp(data, sample_rate, title=title + "_raw", savefig=True)
# 전체의 범위에서 음의 세기의 변화를 MY_SAMPLE_RATE만큼 샘플레이트를 단순화 하여 보여줍니다.
paintbox.show_amp(simplified_abs_data, MY_SAMPLE_RATE, title=title, savefig=True)


# 일부를 추출하여 FFT를 적용해 보겠습니다.
# 원래 데이터에서 적당한 길이를 가져옵니다.
part = data[value * 100 : value * 150]

# FFT를 적용합니다. fft함수는 fft로 구한 값을 (주파수 목록, 값)의 형태로 반환합니다. 옵션을 주지 않으면 4000Hz 이내로만 가져옵니다.
fftbox = toolbox.fft(part, sample_rate)
# FFT 결과를 출력합니다.
paintbox.show_freq(fftbox, title=title, savefig=True)
# 연속적인 값에 가까운 FFT 결과를 음악에서 사용하는 음(musical pitch)으로 이산화 합니다. 
ratiobox = toolbox.compressor(fftbox, setting.HZBORDER)
# 결과를 출력합니다.
paintbox.show_ratio(ratiobox[1], setting.HZLIST, title=title, savefig=True)


# 곡 전체에 대하여 어떤 주파수 대가 많이 나왔는지 2차원 그래프에 그려보겠습니다.
# 곡의 정보를 담고 있는 data를 value 단위로 잘라 2차원으로 만듭니다.
stack = toolbox.stacker_1d(data, value)
print("stack.shape:", stack.shape) # stack은 value단위의 음악 데이터를 가지고 있습니다.

raw_density_map = [] # value단위로 fft한 결과가 담깁니다.
density_map = []     # value단위로 fft하고 음(pitch)로 이산화한 결과가 담깁니다.

for i in range(stack.shape[0]):
    # fft를 수행한 후 raw_density_map에 추가합니다.
    fftbox = toolbox.fft(stack[i], sample_rate)
    raw_density_map.append(fftbox[1])
    # 음(pitch)로 이산화한 후 density_map에 추가합니다
    ratiobox = toolbox.compressor(fftbox, setting.HZBORDER)
    density_map.append(ratiobox[1])

# ndarray로 바꿔줍니다.
raw_density_map = np.array(raw_density_map)
density_map = np.array(density_map)

# 각각의 결과를 보여줍니다.
paintbox.show_density(raw_density_map, MY_SAMPLE_RATE, title=title + "_RAW", savefig=True)
paintbox.show_density(density_map, MY_SAMPLE_RATE, title=title + "_ONPITCH", savefig=True)


# 위에서 구한 값들을 바탕으로 음을 특정 간격으로 구분하는 것에 대해 그래프로 표현해 보겠습니다.
print("simplified_abs_data.shape:", simplified_abs_data.shape)
print("raw_density_map.shape:", raw_density_map.shape)

# byAmp에는 value단위 음의 세기에 대한 정보가 들어있는 simplified_abs_data를 15단위에서 부터 900단위까지 구분하였을 때 유사도에 대한 정보가 담깁니다.
# MY_SAMPLE_RATE가 30이므로 15는 0.5초 900은 30초 입니다.
byAmp = toolbox.term_cal_1d(simplified_abs_data, 15, 900)
# 단위를 초(sec)로 변경
byAmp[:,0] = byAmp[:, 0] / MY_SAMPLE_RATE

# byFreq에는 value단위 음의 주파수에 대한 정보가 들어있는 raw_density_map을 15단위에서 부터 900단위까지 구분하였을 때 유사도에 대한 정보가 담깁니다.
# MY_SAMPLE_RATE가 30이므로 위와 같이 15는 0.5초 900은 30초 입니다.
byFreq = toolbox.term_cal_2d(raw_density_map, 15, 900)
# 단위를 초(sec)로 변경
byFreq[:, 0] = byFreq[:, 0] / MY_SAMPLE_RATE

paintbox.show_fast_2d(byAmp, xlabel="Term(sec)", title=title + "_Long(byAmp)", savefig=True)
paintbox.show_fast_2d(byFreq, xlabel="Term(sec)", title=title + "_Long(byFreq)", savefig=True)


# 위와 동일한 방법으로 짧은 범위에서
byAmp = toolbox.term_cal_1d(simplified_abs_data, 15, 90)
byAmp[:,0] = byAmp[:, 0] / MY_SAMPLE_RATE

byFreq = toolbox.term_cal_2d(raw_density_map, 15, 90)
byFreq[:, 0] = byFreq[:, 0] / MY_SAMPLE_RATE

paintbox.show_fast_2d(byAmp, xlabel="Term(sec)", title=title + "_Short(byAmp)", savefig=True)
paintbox.show_fast_2d(byFreq, xlabel="Term(sec)", title=title + "_Short(byFreq)", savefig=True)









import numpy as np
from scipy import fftpack

'''
샘플레이트 sample_rate를 가지는 pcm 형태의 data에 대하여 
target_sample_rate를 가지는 pcm형태의 numpy array를 return
Ex. 44100Hz 음원 정보를 단위로 평균내서 60Hz로 단순화
= INPUT => data: numpy_array, sample_rate: int, target_sample_rate: int
= OUTPUT=> numpy_array
'''
def simplifier(data, value, abs=False):
    return_size = len(data) // value
    
    return_data = np.zeros(return_size)
    
    if(abs):
        data = np.abs(data)
    
    # 남으면 그냥 버림
    for i in range(return_size):
        return_data[i] = data[i * value : (i + 1) * value].mean()
    
    return return_data

'''
1차원 ndarray data를 value 단위로 잘라서 쌓아 2차원으로 만든 뒤 반환
= INPUT => data: numpy_array, value: int
= OUTPUT=> numpy_array(?, ?)
'''
def stacker_1d(data, value):
    rest = data.shape[0] % value
    
    # 남으면 버림
    if not rest == 0:
        data = data[:-rest]
    
    return_data = data.reshape(data.shape[0] // value, value)
    return return_data

'''
2차원 ndarray data를 value 단위로 잘라서 3차원으로 만든 뒤 반환
= INPUT => data: numpy_array(?, ?), value: int
= OUTPUT=> numpy_array(?, ?, ?)
'''
def stacker_2d(data, value):
    rest = data.shape[0] % value
    
    # 남으면 버림
    if not rest == 0:
        data = data[:-rest]
    
    return_data = data.reshape(data.shape[0] // value, value, data.shape[-1])
    return return_data

'''
샘플레이트 sample_rate를 가지는 pcm 형태의 data에 대하여
fft 수행하여 (주파수, 해당 값)의 배열로 반환
fullrange가 True이면 주파수 범위 전부, False(default)이면 4000Hz 이내만
= INPUT => data: numpy_array, sample_rate: int
= OUTPUT=> (2, 최대 주파수) 형태의 numpy_array
'''
def fft(data, sample_rate, fullrange=False):
    spectrum = fftpack.fft(data) # fast fourier transform
    freq = fftpack.fftfreq(len(data), d = 1.0 / sample_rate) # spectrum에 맞춰줄 값
    
    spectrum = abs(spectrum[:len(freq) // 2]) / sample_rate
    freq = freq[:len(freq) // 2]
    
    # 4000 Hz 이내만 다룸
    if not fullrange:
        boundary = np.searchsorted(freq, 4000)
        spectrum = spectrum[:boundary]
        freq = freq[:boundary]
    
    return np.vstack((freq, spectrum))

'''
fft 결과물을 받아 이산화 시키는 작업
경계에 대한 정보는 hzboarder를 통해 얻는다. 저 간격 사이에 해당하는 주파수들을 값을 더한다.
= INPUT => data: numpy_array(2, ?), hzboarder: list
= OUTPUT=> numpy_array(2, ?)
'''
def compressor(data, hzboarder, cutoff=-1):
    freq = data[0]
    spectrum = data[1]
    
    if (cutoff == -1):
        cutoff = spectrum.max() * 0.005
    spectrum[spectrum < cutoff] = 0
    
    step = []
    for i in range(len(hzboarder)):
        step.append(np.searchsorted(freq, hzboarder[i]))
    
    splist = np.split(spectrum, step)[1:len(hzboarder) + 1]
    
    if(len(splist) != len(hzboarder)):
        print("Error in 'if(len(splist) == len(hzList_flatten))'!")
        print("len(splist):", len(splist), ", len(hzList_flatten):", len(hzboarder))
    
    sumArr = np.array([arr.sum() for arr in splist])
    if not sumArr.sum() == 0:
        sumArr = sumArr / sumArr.sum()
    
    return np.vstack((hzboarder, sumArr))

'''
입력된 arr에 대하여 start 값 부터 end 값까지 각 단위로 쪼개었을 때 유사도를 계산하여 출력한다.
유사도는 동일한 위치에 있는 값들의 분산값으로 결정한다.
= INPUT => arr: numpy_array(?), start: int, end: int
= OUTPUT=> numpy_array(?, ?)
'''
def term_cal_1d(arr, start, end, cutoff=-1, normalize=False):
    lst = []
    
    # cutoff 값으로 받은것이 있으면 이하의 값은 0으로
    if cutoff != -1:
        arr[arr < cutoff] = 0.0
    
    for intrv in range(start, end + 1):
        temp_arr = stacker_1d(arr, intrv)
        
        if normalize:
            # 각 행을 최대값에 대하여 정규화 (최대값을 1에 맞추어 조정)
            temp_arr = np.divide(temp_arr.transpose(), temp_arr.max(axis = 1)).transpose()
        
        # 모든 행이 얼마나 유사한지 알아보는 과정
        # 모든 행을 대상으로 같은 열 번호에 있는 값들의 분산을 계산하여 평균
        temp = temp_arr.var(axis = 0)
        temp = temp.mean()
        
        lst.append((intrv, temp))
    return np.array(lst)

'''
term_cal_1d의 확장 버전.
= INPUT => arr: numpy_array(?, ?), start: int, end: int
= OUTPUT=> numpy_array(?, ?)
'''
def term_cal_2d(arr, start, end, cutoff=-1):
    lst = []
    
    # cutoff 값으로 받은것이 있으면 이하의 값은 0으로
    if cutoff != -1:
        arr[arr < cutoff] = 0.0
    
    for intrv in range(start, end + 1):
        temp_arr = stacker_2d(arr, intrv)
        
        # 모든 행이 얼마나 유사한지 알아봄
        temp = temp_arr.var(axis = 0)
        temp = temp.sum(axis = 1)
        temp = temp.mean()
        
        lst.append((intrv, temp))
    return np.array(lst)

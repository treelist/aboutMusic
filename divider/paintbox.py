
import numpy as np
import matplotlib.pyplot as plt
import datetime

'''
data 그려주기.
savefig가 True이면 저장, False(default)이면 보여주기만.
= INPUT => data: numpy_array, sample_rate: int
'''
def show_amp(data, sample_rate, title="", savefig=False):
    plt.clf() # clear figure, 그냥
    
    if savefig:
        fig = plt.figure(figsize=(15, 15))
    
    plt.xlabel("time [s]") # x축의 단위는 초
    plt.title(title + "[Amplitude]") # 표 이름
    
    plot_x = np.arange(len(data)) / sample_rate # (1 / sample_rate)초 단위
    plot_y = data
    plt.plot(plot_x, plot_y)
    
    plt.grid(True) # 보기 좋음
    
    if savefig:
        now = datetime.datetime.now().strftime("%H%M%S")
        plt.savefig("[" + now + "]" + title + "(Amplitude).png")
    else:
        plt.show()

'''
fft 결과 출력용.
= INPUT => data: numpy_array(2, ?), sample_rate: int
'''
def show_freq(data, title="", savefig=False):
    plt.clf()
    
    if savefig:
        fig = plt.figure(figsize=(15, 15))
    
    plt.xlabel("frequency [Hz]") # 단위는 주파수
    plt.title(title + "[Frequency]")
    
    plot_x = data[0]
    plot_y = data[1]
    plt.plot(plot_x, plot_y, linewidth = 2)
    
    plt.grid(True) # 보기 좋음
    
    if savefig:
        now = datetime.datetime.now().strftime("%H%M%S")
        plt.savefig("[" + now + "]" + title + "(Frequency).png")
    else:
        plt.show()

'''
= INPUT => hzlist: list(int), start: int, gap: int
= OUTPUT=> list(str)
'''
def ticks_maker(hzlist, start, gap):
    xticks = [" "] * len(hzlist)
    
    i = start
    while (i < len(hzlist)):
        xticks[i] = str(hzlist[i])
        i += gap

    return xticks

'''
frequency 간격별로 출력용.
= INPUT => data: numpy_array, hzlist: list(int)
'''
def show_ratio(data, hzlist, title="", savefig=False):
    plt.clf()
    
    if savefig:
        fig = plt.figure(figsize=(15, 15))
    
    plt.xlabel("frequency [Hz]")
    plt.title(title + "[Ratio]")
    
    xticks = ticks_maker(hzlist, 11, 12)
    
    plt.xticks(range(len(data)), xticks)
    plt.xlim(hzlist[0], len(data))
    
    bar_x = range(len(data))
    bar_y = data
    plt.bar(bar_x, bar_y, width = 1.0)
    
    if savefig:
        now = datetime.datetime.now().strftime("%H%M%S")
        plt.savefig("[" + now + "]" + title + "(Ratio).png")
    else:
        plt.show()

'''
2차원 맵 그리기.
입력값 data는 단위 간격당 pitch값을 가진 음원에 대한 데이터
= INPUT => data: numpy_array(?, ?), hzlist: list(int)
'''
def show_density(data, sample_rate, title="", savefig=False):
    plt.clf()
    
    if savefig:
        fig = plt.figure(figsize=(15, 15))
    else:
        fig = plt.figure(figsize=(6, 6))
    
    x = np.linspace(0, data.shape[0], data.shape[0] + 1) / sample_rate
    y = np.linspace(0, data.shape[1], data.shape[1])
    
    
    x, y = np.meshgrid(x, y, indexing='ij')
    
    plt.xlabel("time [s]")
    #plt.ylabel("scale is wrong")
    plt.tick_params(axis="y", which="both", top="off", right="off", left="off", bottom="off", labeltop="off", labelright="off", labelleft="off", labelbottom="off")
    plt.pcolormesh(x, y, data, cmap="Blues")
    plt.colorbar()

    if savefig:
        now = datetime.datetime.now().strftime("%H%M%S")
        plt.savefig("[" + now + "]" + title + "(FrequencyMap).png")
    else:
        plt.show()

'''
2차원 ndarray에 대한 빠른 확인용.
= INPUT => data: numpy_array(?, ?)
'''
def show_fast_2d(arr, xlabel="", ylabel="", title="", savefig=False):
    plt.clf()
    
    if savefig:
        fig = plt.figure(figsize=(15, 15))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(arr[:,0], arr[:,1])
    
    if savefig:
        now = datetime.datetime.now().strftime("%H%M%S")
        plt.savefig("[" + now + "]" + title + ".png")
    else:
        plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
df = pd.read_csv('/home/user/Desktop/dawn-deeplearning-ECG-usingMat/2022-05-27 03:47:01.523784.csv')
ecg = df.values.reshape(len(df))
ecgre = scipy.signal.resample(ecg,5000)

import numpy as np

# from scipy.signal import savgol_filter
# ecgre = savgol_filter(ecgre, 51, 3)

import logging, time
from biosppy.signals import ecg
import sys
fs = 360  # 信号采样率 360 Hz

tic = time.time()
rpeaks = ecg.christov_segmenter(ecgre, sampling_rate=fs)

print(rpeaks[0][1])
wave =[]
for i in range(len(rpeaks[0])):
  
    wave.append(ecgre[rpeaks[0][i]-100:rpeaks[0][i]+150])
 


from tensorflow.keras import models
model = models.load_model("DNN_new.h5")
#model.summary()
import matplotlib.pyplot as plt
for i in range(1):
    print(model.predict_classes(wave[i].reshape(1,250)))   
    plt.plot(wave[i])
plt.show()
toc = time.time()
logging.info("完成. 用时: %f 秒. " % (toc - tic))
# 以上这种方式返回的rpeaks类型为biosppy.utils.ReturnTuple, biosppy的内置类
logging.info("直接调用 christov_segmenter 返回类型为 " + str(type(rpeaks)))

# 得到R波位置序列的方法：
# 1) 取返回值的第1项：
logging.info("使用第1种方式取R波位置序列 ... ")
rpeaks_indices_1 = rpeaks[0]
logging.info("完成. 结果类型为 " + str(type(rpeaks_indices_1)))
# 2) 调用ReturnTuple的as_dict()方法，得到Python有序字典（OrderedDict）类型
logging.info("使用第2种方式取R波位置序列 ... ")
rpeaks_indices_2 = rpeaks.as_dict()
#    然后使用说明文档中的参数名（这里是rpeaks）作为key取值。
rpeaks_indices_2 = rpeaks_indices_2["rpeaks"]
logging.info("完成. 结果类型为 " + str(type(rpeaks_indices_2)))

# 检验两种方法得到的结果是否相同：
check_sum = np.sum(rpeaks_indices_1 == rpeaks_indices_2)
if check_sum == len(rpeaks_indices_1):
    logging.info("两种取值方式结果相同 ... ")
else:
    logging.info("两种取值方式结果不同，退出 ...")
    sys.exit(1)

# 与 christov_segmenter 接口一致的还有 hamilton_segmenter
logging.info("调用接口一致的 hamilton_segmenter 进行R波检测")
tic = time.time()
rpeaks = ecg.hamilton_segmenter(ecgre, sampling_rate=fs)
toc = time.time()
logging.info("完成. 用时: %f 秒. " % (toc - tic))
rpeaks_indices_3 = rpeaks.as_dict()["rpeaks"]
# 绘波形图和R波位置
num_plot_samples = 3600
logging.info("绘制波形图和检测的R波位置 ...")
sig_plot = ecgre[:num_plot_samples]
rpeaks_plot_1 = rpeaks_indices_1[rpeaks_indices_1 <= num_plot_samples]
plt.figure()
plt.plot(sig_plot, "g", label="ECG")
plt.grid(True)
plt.plot(rpeaks_plot_1, sig_plot[rpeaks_plot_1], "ro", label="christov_segmenter")
rpeaks_plot_3 = rpeaks_indices_3[rpeaks_indices_3 <= num_plot_samples]
plt.plot(rpeaks_plot_3, sig_plot[rpeaks_plot_3], "b^", label="hamilton_segmenter")
plt.legend()
plt.title("2022-05-25 02:48:08.944683")
plt.show()
logging.info("完成.")


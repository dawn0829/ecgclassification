Path=r'/home/user/Desktop/dawn-deeplearning-ECG-usingMat/MITBIH_Mat' #自定义路径要正确
DataFile='/Data_CNN.mat'
LabelFile='/Label_OneHot.mat'
DataFile1='/N_dat.mat'
 
from cProfile import label
import h5py as hp
import numpy as np
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data,'r')
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr
from sklearn.utils import resample 
import pandas as pd
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T #TransferMat in ndarray

label_df,data_df = [],[]
df_1_upsample=resample(Data[0:5000],replace=True,n_samples=20000)
for i in range(20000):
    label_df.append([1,0,0,0])
    data_df.append(df_1_upsample[i])

df_2_upsample=resample(Data[5000:10000],replace=True,n_samples=20000)
for i in range(20000):
    label_df.append([0,1,0,0])
    data_df.append(df_2_upsample[i])
import matplotlib.pyplot as plt

df_3_upsample=resample(Data[10000:15000],replace=True,n_samples=20000)
for i in range(20000):
    label_df.append([0,0,1,0])
    data_df.append(df_3_upsample[i])
df_4_upsample=resample(Data[15000:20000],replace=True,n_samples=20000)
for i in range(20000):
    label_df.append([0,0,0,1])
    data_df.append(df_4_upsample[i])

Indices=np.arange(len(data_df)) #rearange Data Index
print(len(data_df),len(label_df))
np.random.shuffle(Indices)

#print(Data[0])
print(Indices[:70000])

train_x=np.array(data_df)[Indices[:60000]]
train_y=np.array(label_df)[Indices[:60000]]
test_x=np.array(data_df)[Indices[60000:75000]]
test_y=np.array(label_df)[Indices[60000:75000]]
val_x=np.array(data_df)[Indices[75000:]]
val_y=np.array(label_df)[Indices[75000:]]

from tensorflow.keras.models import Sequential  #← 匯入 Keras 的序列式模型類別
from tensorflow.keras.layers import Dense       #← 匯入 Keras 的密集層類別
from tensorflow.keras import optimizers
model = Sequential()                 #← 建立序列模型物件
model.add(Dense(128, activation='relu', input_dim= 250)) #← 加入第一層
model.add(Dense(4, activation='softmax'))               #← 加入第二層

opt = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=opt,             #← 指定優化器
              loss='categorical_crossentropy', #← 指定損失函數
              metrics=['acc'])                 #← 指定評量準則

#history = model.fit(train_x, train_y, epochs=5, batch_size=64)
#print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
#程 訓練模型
model.summary()
history = model.fit(train_x, train_y, epochs=5, batch_size=128)

#程 評估模型成效
test_loss, test_acc = model.evaluate(test_x, test_y)   #←使用測試樣本及標籤來評估普適能力
print('對測試資料集的準確率：', test_acc)

#程 畫出測試圖片並標示預測結果與標準答案

predict = model.predict_classes(test_x)  #←用測試樣本進行預測
print(test_y)
print(predict)
#程 將模型存檔
model.save('DNN_new.h5')   #← 將模型以指定的檔名存檔


#程 評估模型成效
test_loss, test_acc = model.evaluate(val_x, val_y)   #←使用測試樣本及標籤來評估普適能力
print('對測試資料集的準確率：', test_acc)

#程 畫出測試圖片並標示預測結果與標準答案

predict = model.predict_classes(test_x)  #←用測試樣本進行預測
print(test_y)
print(predict)
#程 將模型存檔
model.save('DNN_new1.h5')   #← 將模型以指定的檔名存檔
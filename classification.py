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

# for i in range(2000):
#     plt.plot(Data[i])
#     print(Label[i])
# plt.show()
# for i in range(2000):
#     plt.plot(Data[i+5000])
#     print(Label[i+5000])
# plt.show()
# for i in range(2000):
#     plt.plot(Data[i+10000])
#     print(Label[i+10000])
# plt.show()
# for i in range(2000):
#     plt.plot(Data[i+15000])
#     print(Label[i+15000])
# plt.show()

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

# import matplotlib.pyplot as plt
# for i in range(1):
#     plt.plot(train_x[i+1002])
#     print(train_y[i+1002])
# plt.show()

train_x = np.array(train_x).reshape((60000,250,1))

train_y = np.array(train_y).reshape((60000,4))

test_x = np.array(test_x).reshape((15000,250,1))

test_y = np.array(test_y).reshape((15000,4))

val_x = np.array(val_x).reshape((5000,250,1))

val_y = np.array(val_y).reshape((5000,4))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import callbacks
from tensorflow import keras
def network(X_train,y_train,X_test,y_test):
    im_shape=(X_train.shape[1],1)

    cnn = Sequential()
    cnn.add(Conv1D(64, (3), activation='relu',input_shape=(im_shape))) 
    cnn.add(MaxPooling1D(2))
    cnn.add(Dropout(0.25))
    cnn.add(Conv1D(64, (3), activation='relu'))
    cnn.add(MaxPooling1D(2))
    cnn.add(Dropout(0.25))
    cnn.add(Conv1D(64, (3), activation='relu'))
    cnn.add(MaxPooling1D(2))
    cnn.add(Flatten()) 
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(4, activation='softmax'))
    
    opt = keras.optimizers.Adam(learning_rate=0.00001)

    # -- 神經網路的訓練配置 -- #
    cnn.compile(loss='categorical_crossentropy',	# 損失函數
              optimizer=opt,				    # adam優化器
              metrics=['acc'])			    # 以準確度做為訓練指標
    
    callback = [callbacks.EarlyStopping(monitor='val_loss', patience=8),
             callbacks.ModelCheckpoint(filepath='testvaldata_model.h5', monitor='val_loss', save_best_only=True)]

    history = cnn.fit(X_train, y_train, epochs=35, callbacks=callback, batch_size=128, validation_data=(val_x,val_y))

    return(cnn,history)


model,history = network(train_x,train_y,test_x,test_y)

for i in range(10):
    y = model.predict_classes(test_x[i].reshape((1, 250, 1)))
    print(y)
    print(test_y[i])
print("Evaluate on test data")
results = model.evaluate(test_x,test_y, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_x[:3])
print("predictions shape:", predictions.shape)

# import matplotlib.pyplot as plt
# for i in range(100):
       
# from tensorflow.keras.models import Sequential  #← 匯入 Keras 的序列式模型類別
# from tensorflow.keras.layers import Dense       #← 匯入 Keras 的密集層類別

# model = Sequential()                 #← 建立序列模型物件
# #model.add(Dense(128, activation='relu', input_dim= 250)) #← 加入第一層
# model.add(Dense(4, activation='softmax'))               #← 加入第二層
# model.compile(optimizer='rmsprop',             #← 指定優化器
#               loss='categorical_crossentropy', #← 指定損失函數
#               metrics=['acc'])                 #← 指定評量準則

# #history = model.fit(train_x, train_y, epochs=5, batch_size=64)
# #print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
# #程 訓練模型
# model.summary()
# history = model.fit(train_x, train_y, epochs=5, batch_size=128)

# #程 評估模型成效
# test_loss, test_acc = model.evaluate(test_x, test_y)   #←使用測試樣本及標籤來評估普適能力
# print('對測試資料集的準確率：', test_acc)

# #程 畫出測試圖片並標示預測結果與標準答案

# predict = model.predict_classes(test_x)  #←用測試樣本進行預測
# print(test_y)
# print(predict)
# #程 將模型存檔
# model.save('DNN_new1.h5')   #← 將模型以指定的檔名存檔
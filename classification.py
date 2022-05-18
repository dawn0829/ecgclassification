Path=r'/home/user/Desktop/dawn-deeplearning-ECG-usingMat/MITBIH_Mat' #自定义路径要正确
DataFile='/Data_CNN.mat'
LabelFile='/Label_OneHot.mat'
 
import h5py as hp
import numpy as np
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data,'r')
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr
 

Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T #TransferMat in ndarray

Indices=np.arange(Data.shape[0]) #rearange Data Index

np.random.shuffle(Indices)

#print(Data[0])
# import matplotlib.pyplot as plt
# for i in range(2000):
#     plt.plot(Data[0:5000])
# plt.show()
train_x=Data[Indices[:15000]]
train_y=Label[Indices[:15000]]
test_x=Data[Indices[15000:]]
test_y=Label[Indices[15000:]]

train_x = np.array(train_x)
train_x = train_x.reshape((15000,250,1))
train_y = np.array(train_y)
train_y = train_y.reshape((15000,4))
test_x = np.array(test_x)
test_x = test_x.reshape((5000,250,1))
test_y = np.array(test_y)
test_y = test_y.reshape((5000,4))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import callbacks
def network(X_train,y_train,X_test,y_test):
    im_shape=(X_train.shape[1],1)

    cnn = Sequential()
    cnn.add(Conv1D(64, (6), activation='relu', padding='same',input_shape=(im_shape))) 
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=(3),strides=(2)))
    cnn.add(Conv1D(64, (3), activation='relu', padding='same',input_shape=(im_shape))) 
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=(2),strides=(2)))
    cnn.add(Conv1D(64, (3), activation='relu', padding='same',input_shape=(im_shape)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=(2),strides=(2)))
    cnn.add(Flatten()) 
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(4, activation='softmax'))
    


    # -- 神經網路的訓練配置 -- #
    cnn.compile(loss='categorical_crossentropy',	# 損失函數
              optimizer='adam',				    # adam優化器
              metrics=['acc'])			    # 以準確度做為訓練指標
    
    callback = [callbacks.EarlyStopping(monitor='val_loss', patience=8),
             callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history = cnn.fit(X_train, y_train, epochs=10, callbacks=callback, batch_size=64, validation_data=(X_test,y_test))

    cnn.load_weights('best_model.h5')
    history = 0
    return(cnn,history)


model,history = network(train_x,train_y,test_x,test_y)

for i in range(10):
    y = model.predict(train_x[i].reshape((1, 250, 1)))
    print(y)
    print(test_y[i])
from tensorflow.keras.models import Sequential  #← 匯入 Keras 的序列式模型類別
from tensorflow.keras.layers import Dense       #← 匯入 Keras 的密集層類別

model = Sequential()                 #← 建立序列模型物件
#model.add(Dense(128, activation='relu', input_dim= 250)) #← 加入第一層
model.add(Dense(4, activation='softmax'))               #← 加入第二層
model.compile(optimizer='rmsprop',             #← 指定優化器
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
model.save('模型_DNN_new.h5')   #← 將模型以指定的檔名存檔
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

# import matplotlib.pyplot as plt
# for i in range(500):
#     plt.plot(Data[i+15000])
# plt.show()

Indices=np.arange(Data.shape[0]) #rearange Data Index

np.random.shuffle(Indices)
train_x=Data[Indices[:19500]]
train_y=Label[Indices[:19500]]
test_x=Data[Indices[19500:]]
test_y=Label[Indices[19500:]]
# wash card test

import tensorflow as tf
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32,activation='relu', input_shape=(250,)))
model.add(tf.keras.layers.Dense(4, activation='softmax'))  
model.compile(optimizer=opt,             #← 指定優化器
              loss='categorical_crossentropy', #← 指定損失函數
              metrics=['acc'])                 #← 指定評量準則

model.summary()
DataFile1 = '/N_dat.mat'
L1=load_mat(Path+DataFile1,'Nb')
L1=L1.T
val_x = L1
val_y = list()
# for i in range(val_x.len()):
#     val_y.append([1,0,0,0])
print(L1.shape)

history = model.fit(train_x, train_y, epochs=100, batch_size=64, validation_data=(test_x,test_y))

test_loss, test_acc = model.evaluate(test_x, test_y)   #←使用測試樣本及標籤來評估普適能力
print('對測試資料集的準確率：', test_acc,test_loss)

# DataFile1 = '/N_dat.mat'
# L1=load_mat(Path+DataFile1,'Nb')
# L1=L1.T

# DataFile1 = '/V_dat.mat'
# L2=load_mat(Path+DataFile1,'Vb')
# L2=L2.T

# DataFile1 = '/L_dat.mat'
# L3=load_mat(Path+DataFile1,'Lb')
# L3=L3.T

# DataFile1 = '/R_dat.mat'
# L4=load_mat(Path+DataFile1,'Rb')
# L4=L4.T
import matplotlib.pyplot as plt
# for i in range(10):
#     print(model.predict(L1[i].reshape(1,250)))   
#     plt.plot(L1[i])
# plt.show()

# for i in range(10):
#     print(model.predict_classes(L2[i].reshape(1,250)))   
#     plt.plot(L2[i])
# plt.show()

# for i in range(10):
#     print(model.predict_classes(L3[i].reshape(1,250)))   
#     plt.plot(L3[i])
# plt.show()

# for i in range(10):
#     print(model.predict_classes(L4[i].reshape(1,250)))   
#     plt.plot(L4[i])
# plt.show()
predict = model.predict_classes(test_x)
plt.gcf().set_size_inches(15,4)
for i in range(5):
    ax = plt.subplot(1,5,1+i)
    ax.plot(test_x[i+400])
    ax.set_title('lable ='+str(test_y[i+400])+'\npredi ='+str(predict[i+400]),fontsize=18)
    ax.set_xticks([]);ax.set_yticks([])
plt.show()


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()
# summarize history for loss
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()

model.save('模型_DNN_new.h5')
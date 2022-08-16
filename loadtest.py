from tensorflow.keras import models
import h5py as hp
import numpy as np
Path=r'/home/user/Desktop/dawn-deeplearning-ECG-usingMat/MITBIH_Mat' #自定义路径要正确
DataFile='/Data_CNN.mat'

def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr
 
L=load_mat(Path+DataFile,'Data')
L=L.T
model = models.load_model("testvaldata_model.h5")
model.summary()


results = model.evaluate(test_x,test_y, batch_size=128)
# import matplotlib.pyplot as plt
# for i in range(5000):
#     plt.plot(L[i])
#     if(model.predict_classes(L[i].reshape((1, 250, 1))!=[1,0,0,0])):
#         print(L[i])
# plt.show()

# for i in range(5000):
#     plt.plot(L[i+5000])
#     if(model.predict_classes(L[i+5000].reshape((1, 250, 1))!=[0,1,0,0])):
#         print(L[i+5000])
# plt.show()

# for i in range(5000):
#     plt.plot(L[i+10000])
#     if(model.predict_classes(L[i+10000].reshape((1, 250, 1))!=[0,0,1,0])):
#         print(L[i+10000])
# plt.show()

# for i in range(5000):
#     plt.plot(L[i+15000])
#     if(model.predict_classes(L[i+15000].reshape((1, 250, 1))!=[0,0,0,1])):
#         print(L[i+15000])
# plt.show()


from pyexpat import model
from tensorflow.keras import models
import h5py as hp
import numpy as np
Path=r'/home/user/Desktop/dawn-deeplearning-ECG-usingMat/MITBIH_Mat' #自定义路径要正确
DataFile='/N_dat.mat'

def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr
 
L=load_mat(Path+DataFile,'Nb')
L=L.T
model = models.load_model("DNN_new.h5")
#model.summary()
import matplotlib.pyplot as plt
for i in range(100):
    print(model.predict_classes(L[i].reshape(1,250)))   
    plt.plot(L[i])
plt.show()
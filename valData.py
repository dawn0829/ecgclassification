from tensorflow.keras import models
import h5py as hp
import numpy as np
Path=r'/home/user/Desktop/dawn-deeplearning-ECG-usingMat/MITBIH_Mat' #自定义路径要正确
DataFile1='/L_dat.mat'
DataFile2='/N_dat.mat'
DataFile3='/R_dat.mat'
DataFile4='/V_dat.mat'


def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr
 
L=load_mat(Path+DataFile1,'Lb').T
N=load_mat(Path+DataFile2,'Nb').T
R=load_mat(Path+DataFile3,'Rb').T
V=load_mat(Path+DataFile4,'Vb').T

model = models.load_model("testvaldata_model.h5")
#model.summary()
import matplotlib.pyplot as plt
for i in range(100):
    print(model.predict_classes(N[i+1000].reshape(1,250,1)))   
    plt.plot(N[i+1000])
plt.show()


for i in range(100):
    print(model.predict_classes(L[i+1000].reshape(1,250,1)))   
    plt.plot(L[i+1000])
plt.show()


for i in range(100):
    print(model.predict_classes(R[i+1000].reshape(1,250,1)))   
    plt.plot(R[i+1000])
plt.show()

for i in range(100):
    print(model.predict_classes(V[i+1000].reshape(1,250,1)))   
    plt.plot(V[i+1000])
plt.show()
import h5py
import matplotlib.pyplot as plt

filename = './bicycle_dataset.h5'
data = h5py.File(filename, 'r')
for group in data.keys() :
    print (group)
    for dset in data[group].keys():      
        arr = data[group][dset][:] # adding [:] returns a numpy array


plt.scatter(arr[0,:],arr[1,:],s=1)
plt.show()
print("stop")
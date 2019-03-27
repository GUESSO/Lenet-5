import struct
import constant
import numpy as np
def read_image(filename,normalize=False,flatten=True,debug=False):
    f = open(filename,'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, nums_of_images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    input_dim=28*28
    total_data=[]
    if(debug==True):
        nums_of_images=constant.nums_for_debug
    # load data
    for i in range(nums_of_images):
        data=[]
        if flatten:
            for m in range(28):
                rowData = []
                for n in range(28):
                    rowData.append(float(struct.unpack_from('>B', buf, index)[0]))
                    index += struct.calcsize('>B')
                data.append(rowData)
        else:
            for j in range(input_dim):
                data.append(float(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
        data=np.array(data)
        if(normalize):
            # standard normalization
            #
            # mean=np.mean(data)
            # std=np.std(data,ddof=1)
            # data=(data-mean)/std
            data=data/constant.input_max
        total_data.append(data)
    return total_data

def read_label(filename,one_hot=True,debug=False):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, num_of_labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labelArr = []
    if(debug):
        num_of_labels=constant.nums_for_debug
    for x in range(num_of_labels):
        temp=int(struct.unpack_from('>B', buf, index)[0])
        if(one_hot):
            data=np.zeros(10)
            data[int(temp)]=1
        else:
            data=temp
        labelArr.append(data)
        index += struct.calcsize('>B')
    return np.array(labelArr)

def load_MNIST(normalize=False,one_hot=True,flatten=False,debug=False):
    train_data=read_image(constant.train_data_path,normalize,flatten,debug)
    train_label=read_label(constant.train_label_path,one_hot,debug)
    test_data=read_image(constant.test_data_path,normalize,flatten,debug)
    test_label=read_label(constant.test_label_path,one_hot,debug)
    return train_data,train_label,test_data,test_label
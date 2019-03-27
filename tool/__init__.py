# -*- coding: utf-8 -*
import numpy as np
import constant
import datetime
class Tanh:
    def forward(self,x):
        return np.tanh(x).tolist()

    def backward(self,x):
        temp=np.tanh(x)
        return (1-temp**2).tolist()

class Sigmoid:
    def forward(self,x):
        out=1.0 / (1.0 + np.exp(-x))
        self.out=out
        return out

    def backward(self,dout):
        return dout*(1-self.out)*self.out

class ReLU:
    def __init__(self):
        self.mask=None

    def forward(self,x):
        x=np.array(x)
        self.mask=np.where(x>0,1,0)
        return np.where(x > 0, x, 0).tolist()

    def backward(self,dout):
        dout=np.array(dout)*self.mask
        return dout.tolist()

class softmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None

    def forward(self,x,t):
        self.t=t
        self.y=self.softmax(x)
        return self.cross_entropy_loss(self.softmax(x),t)

    def backward(self,dout=1):
        dx=self.y-self.t
        return dx

    def softmax(self,x):
        c=np.max(x)
        exp_a=np.exp(x-c)
        sumExpa=np.sum(exp_a)
        y=exp_a/sumExpa
        return y

    def cross_entropy_loss(self,y, t):
        return -np.sum(t * np.log(y + constant.constantMin))

# structure of input_data
# input_data:
#   channel
#   x:height of image
#   y:width of image
#   data:image_data of pixels
class ConvData:
    def __init__(self,channel,x,y,data):
        self.channel=channel
        self.x=x
        self.y=y
        self.data=data

    def __del__(self):
        del self.x,self.y,self.data,self.channel

def im2col(input_data,filter_w,stride=1,pad=0):
    rst_data = []
    temp_x=input_data.x + pad * 2
    temp_y=input_data.y + pad * 2
    outsize = (temp_x - filter_w) / stride + 1
    for k in range(input_data.channel):
        temp_pad_intput=np.pad(np.array(input_data.data[k]).reshape(input_data.x,input_data.y),((pad,pad)),'constant')
        # print input_data.data[k]
        temp_data=[]
        for i in range(outsize):
            for j in range(outsize):
                temp_i=i*stride
                temp_j=j*stride
                temp=temp_pad_intput[temp_i:temp_i+filter_w]
                temp=temp[:,temp_j:temp_j+filter_w]
                temp_data.append(temp.flatten())
        rst_data.append(np.array(temp_data))
    input_data.x = temp_x
    input_data.y = temp_y
    output_data=ConvData(input_data.channel,input_data.x,input_data.y,np.array(rst_data))
    return output_data

def col2im(self):
        # remain to be done
        return 0

def mean_squared_loss(t,y):
    batch_size=y.shape[0]
    return 0.5*np.sum((y-t)**2)/batch_size

# rotate matrix clockwise
def rotate(matrix):
    return map(list, zip(*matrix[::-1]))

def add_matrix_in_list(matrix_list):
    rst=0
    for i in range(len(matrix_list)):
        rst+=np.array(matrix_list[i])
    return rst

def list_to_str(list_var):
    return '['+','.join(str(e) for e in list_var)+']'

def get_now_time():
    return datetime.datetime.now().strftime('%Y_%m_%d#%H_%M_%S')
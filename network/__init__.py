import numpy as np
import constant
class Neuron:
    def __init__(self,M,N,lr):
        self.lr=lr
        self.M=M
        self.N=N
        # Xavier init
        # self.W=np.random.rand(self.M,self.N)*constant.Weight_init_std/np.sqrt(M/2)
        # HE init
        self.W=np.random.normal(0,np.sqrt(2.0/M),(self.M,self.N))
        self.b=np.zeros(N)
        self.dW=None
        self.db=None
        self.x=None
        self.x_shape=None
        # for AdaGrad
        self.h=None

    def update_lr(self,new_lr):
        self.lr=new_lr

    def forward(self,x):
        self.x=np.array(x)
        y=np.dot(x,self.W)+self.b
        return y

    # what's dout shape,it's(1,N)
    def backward(self,dout):
        self.dout=np.array(dout)
        dx=np.dot(self.dout,self.W.T)
        # reshape(self.M, self.dout.shape[1])
        self.dW=np.dot(self.x.T,self.dout)
        self.db=np.sum(dout,axis=0)
        return dx

    def update(self):
        # for SGD
        self.W=self.W-np.dot(self.dW,self.lr)
        self.b=self.b-np.dot(self.db,self.lr)
        # for AdaGrad
        # self.W=self.W-np.dot(self.dW,self.lr)/(np.sqrt(self.dW**2+constant.constantMin))
        # self.b=self.b-np.dot(self.db,self.lr)
import numpy as np
import constant
class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr

    def update(self,params,grads):
        for x in params.keys():
            params[x]-=grads[x]*self.lr

class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None

    def update(self,params,grads):
        if self.v==None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)

            for key in params.keys():
                self.v[key]=self.momentum*self.v[key]-self.lr*grads[key]
                params[key]+=self.v[key]

class AdaGrad:
    def __init__(self,lr):
        self.lr=lr
        self.h=None

    def update(self,params,grads):
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)

        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key]+constant.constantMin))
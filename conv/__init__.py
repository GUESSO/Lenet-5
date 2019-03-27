import numpy as np
import constant
import tool
class Pos:
    def __init__(self,x,y):
        self.x=x
        self.y=y

# structure for pooling input_data
# first_outside len = num_of_kernel
# second_outside len = channel
# last_outside len = data

class MaxPooling:
    def __init__(self,kernel_size,nums_of_kernels=1):
        self.kernel_size=kernel_size
        self.nums_of_kernels=nums_of_kernels
        self.output_pos=[]
        self.input_size=None
        self.input_data=None
        self.pooling_rst=None
        self.output_size=None
        self.pad=0

    def forward(self,input_data,input_size):
        # self.input_data = input_data
        self.input_size = input_size
        stride = self.kernel_size
        self.output_size = input_size / stride
        self.output_pos=[]
        output_data=[]
        for k in range(self.nums_of_kernels):
            if(len(input_data[k])!=1):
                temp=[]
                temp.append(input_data[k])
                input_data[k]=temp

            pool_data=tool.ConvData(1, self.input_size, self.input_size, input_data[k])
            temp_data=tool.im2col(pool_data,self.kernel_size,stride,0)
            temp_output_data=[]
            temp_max_pos=[]
            temp_data.data=temp_data.data[0]
            # print temp_data.data
            length=self.output_size**2
            for i in range(length):
                row=i/self.output_size*self.kernel_size
                col=i%self.output_size*self.kernel_size
                max_index=np.argmax(temp_data.data[i])
                max_pos=Pos(row+max_index/self.kernel_size,col+max_index%self.kernel_size)
                temp_output_data.append(np.max(temp_data.data[i]))
                temp_max_pos.append(max_pos)
            self.output_pos.append(temp_max_pos)
            output_data.append(np.array(temp_output_data).reshape(self.output_size,-1))
            self.pooling_rst=output_data
        return output_data

    def backward(self,dout):
        output=[]
        for k in range(self.nums_of_kernels):
            output_data = np.zeros((self.input_size, self.input_size))
            temp=dout[k].flatten()
            for i in range(len(self.output_pos[k])):
                pos=self.output_pos[k][i]
                output_data[int(pos.x)][int(pos.y)]=temp[i]
            output.append(output_data)
        return output

class MeanPooling:
    def __init__(self,kernel_size,nums_of_kernels=1):
        self.kernel_size=kernel_size
        self.nums_of_kernels=nums_of_kernels
        self.input_size=None
        self.input_data=None
        self.pooling_rst=None
        self.output_size=None
        self.pad=0

    def forward(self,input_data,input_size):
        self.input_size = input_size
        stride = self.kernel_size
        self.output_size = input_size / stride
        output_data=[]
        for k in range(self.nums_of_kernels):
            if(len(input_data[k])!=1):
                temp=[]
                temp.append(input_data[k])
                input_data[k]=temp

            pool_data=tool.ConvData(1, self.input_size, self.input_size, input_data[k])
            temp_data=tool.im2col(pool_data,self.kernel_size,stride,0)
            temp_output_data=[]
            temp_data.data=temp_data.data[0]
            # print temp_data.data
            length=self.output_size**2
            for i in range(length):
                temp_output_data.append(np.mean(temp_data.data[i]))
            output_data.append(np.array(temp_output_data).reshape(self.output_size,-1))
            self.pooling_rst=output_data
        return output_data

    def backward(self,dout):
        output=[]
        nums_of_one_block=self.kernel_size**2
        for k in range(self.nums_of_kernels):
            output_data = np.zeros((self.input_size, self.input_size))
            temp_len=np.array(dout[k]).flatten()
            for i in range(len(temp_len)):
                row=i/self.output_size*self.kernel_size
                col=i%self.output_size*self.kernel_size
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        output_data[row+m][col+n]=temp_len[i]/nums_of_one_block
            output.append(output_data)
        return output

class inputLayer:
    def __init__(self,target_size):
        self.target_size=target_size

    def forward(self,data):
        input_size=len(data)
        self.data=data
        if(self.target_size!=input_size):
            pad=(self.target_size-input_size)/2
            return np.pad(self.data, ((pad, pad)), 'constant')

class Convolution_3D:
    # nums_of_kernel remain to be done
    def __init__(self,channel,kernel_size,nums_of_kernel,lr):
        self.x=None
        self.lr=lr
        self.input_size=None
        self.output_size=None
        self.channel=channel
        self.kernel_size=kernel_size
        self.nums_of_kernel=nums_of_kernel
        self.W = []
        # for AdaGrad
        self.h=None
        HE_constant=self.kernel_size**2*self.nums_of_kernel*self.channel
        for i in range(nums_of_kernel):
            # self.W.append(np.random.randn(self.channel,kernel_size, kernel_size) * constant.Weight_init_std)
            w_for_channel=[]
            for j in range(self.channel):
                # w_for_channel.append(np.random.normal(0,0.1,(self.kernel_size,self.kernel_size)))
                w_for_channel.append(np.random.normal(0,np.sqrt(2.0/HE_constant),(self.kernel_size,self.kernel_size)))
            self.W.append(np.array(w_for_channel))
        self.dataT=None
        self.dw=None

    def forward(self,data,input_size,stride=1,pad=0):
        self.data=[]
        self.dataT=[]
        self.input_size = input_size
        self.output_size = (input_size - self.kernel_size) / stride + 1
        self.stride = stride
        self.pad = pad
        for i in range(self.channel):
            temp_dT = np.array(data[i])
            self.data.append(temp_dT)
            self.dataT.append(temp_dT.T)

        output_data=[]
        input_data = tool.ConvData(self.channel, input_size, input_size, data)
        input_data = tool.im2col(input_data, self.kernel_size, stride, pad)
        for k in range(self.nums_of_kernel):
            temp_output=[]
            for i in range(self.channel):
                temp_w=self.W[k][i].flatten()
                temp_d=np.dot(input_data.data[i], temp_w).reshape(self.output_size,-1)
                temp_output.append(temp_d)
            output_data.append(tool.add_matrix_in_list(temp_output))
        return output_data

    # dout size equals to conv output size
    def backward(self,dout):
        self.dout = dout
        self.dw=[]
        grad_data = tool.ConvData(self.channel, self.input_size, self.input_size, self.data)
        grad_data = tool.im2col(grad_data, self.output_size, self.stride, 0)
        for k in range(self.nums_of_kernel):
            temp_grad = np.array(dout[k]).flatten()
            dw_temp=[]
            for i in range(self.channel):
                temp_output = np.dot(grad_data.data[i], temp_grad).reshape((self.kernel_size,self.kernel_size))
                dw_temp.append(temp_output)
            # self.dw.append(tool.add_matrix_in_list(dw_temp))
            self.dw.append(dw_temp)

        dx_temp = []
        dout_data = tool.ConvData(1, self.output_size, self.output_size, dout)
        dout_data = tool.im2col(dout_data, self.kernel_size, self.stride, self.kernel_size - 1)
        for i in range(self.channel):
            temp_for_kernel=[]
            for k in range(self.nums_of_kernel):
                temp_w = tool.rotate(self.W[k][i])  # rotate 90 degree
                temp_w = np.array(tool.rotate(temp_w)).flatten()  # rotate 180 degree
                temp_output = np.dot(dout_data.data, temp_w).reshape(self.input_size, -1)
                temp_for_kernel.append(temp_output)
            dx_temp.append(tool.add_matrix_in_list(temp_for_kernel))
        return dx_temp

    def update(self):
        if self.h is None:
            self.h=np.zeros_like(self.W)
        for k in range(self.nums_of_kernel):
            for i in range(self.channel):
                # for SGD
                self.W[k][i]=self.W[k][i]-np.dot(self.dw[k][i],self.lr)

                # for AdaGrad
                # self.h[k][i]+=self.W[k][i]**2
                # self.W[k][i]=self.W[k][i]-self.lr*self.dw[k][i]/(np.sqrt(self.h[k][i]+constant.constantMin))

    def con_format_input(self,data):
        format_rst=[]
        for i in range(self.channel):
            format_rst.append(data)
        return format_rst

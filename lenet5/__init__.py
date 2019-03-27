# -*- coding: utf-8 -*
import dataSet
import conv
import collections
import tool
import numpy as np
import network
import constant
import matplotlib.pyplot as plt
from pylab import *
import time
import json

class lenet5_C3_conv:
    def __init__(self):
        self.C3_conv_layers = []
        self.C3_output = None
        self.connection_index = []
        self.conv_map=np.array(constant.lenet5_c3).T
        self.input_size=14
        self.output_size=10
        for i in range(self.conv_map.shape[0]):
            input_index=[]
            for j in range(self.conv_map.shape[1]):
                if(self.conv_map[i][j]==1):
                    input_index.append(j)
            temp_conv=conv.Convolution_3D(len(input_index),5,1,constant.lr)
            self.connection_index.append(input_index)
            self.C3_conv_layers.append(temp_conv)

    def forward(self,S2_output):
        self.C3_output=[]
        for i in range(len(self.connection_index)):
            ll=[]
            for j in range(len(self.connection_index[i])):
                ll.append(S2_output[j])
            temp_output = self.C3_conv_layers[i].forward(ll, self.input_size, 1, 0)
            # sum to get output of all channels pic
            temp_channel_summed_output = tool.add_matrix_in_list(temp_output)
            self.C3_output.append(temp_channel_summed_output)
        return self.C3_output

    def backward(self,dout):
        dx=[]
        for i in range(len(self.C3_conv_layers)):
            ll=[]
            ll.append(dout[i])
            dx_temp=self.C3_conv_layers[i].backward(ll)
            dx.append(dx_temp)

        dx_for_previous_layer = np.zeros((6, 14, 14))
        for j in range(16):
            index_temp=0
            for i in self.connection_index[j]:
                dx_for_previous_layer[i]+=dx[j][index_temp]
                index_temp+=1


        # dx_for_previous_layer=[]
        # for i in range(6):
        #     temp_add_output=0.0
        #     for j in range(16):
        #     # 16个输出中对应的dx直接加起来即可，就是6个内每个的delta
        #         if(constant.lenet5_c3[i][j]==1):
        #             temp_add_output+=np.array(dx[j])
        #     dx_for_previous_layer.append(temp_add_output)
        return dx_for_previous_layer

    def update(self):
        for i in range(len(self.C3_conv_layers)):
            self.C3_conv_layers[i].update()

class lenet5_C5_conv:
    def __init__(self,channel,kernel_size,nums_of_kernels,lr):
        self.dataT=[]
        self.conv = conv.Convolution(channel,kernel_size,nums_of_kernels,lr)
        self.data=None
        self.cache_dw=[]

    def forward(self,P4_output,input_size):
        C5_output = 0
        self.data=P4_output
        for p4_data in P4_output:
            ll = []
            ll.append(p4_data)
            C5_output += np.array(self.conv.forward(ll, input_size)).reshape(120,)
            self.dataT.append(self.conv.dataT)
        return C5_output

    def backward(self,dout):
        dout_C5 = []
        for i in range(16):
            self.conv.data = self.data[i]
            self.conv.dataT = self.dataT[i]
            temp_dout_C5 = self.conv.backward(dout)
            self.cache_dw.append(self.conv.dw)
            dout_C5.append(temp_dout_C5)
        self.data=None
        self.dataT=[]
        return dout_C5

    def update(self):
        for dw in self.cache_dw:
            self.conv.dw=dw
            self.conv.update()
        self.cache_dw=[]


class lenet5_C:
    def __init__(self,is_debug=False):
        train_data, train_label, test_data, test_label = dataSet.load_MNIST(True, True, True, is_debug)
        self.train_data=train_data
        self.train_label=train_label
        self.test_data=test_data
        self.test_label=test_label
        self.lr=constant.lr
        self.is_debug=is_debug

        self.I0 = conv.inputLayer(32)
        self.C1 = conv.Convolution_3D(1, 5, 6, self.lr)  # channel,kernel_size,nums_of_kernel,lr
        self.relu1=tool.ReLU()
        self.P2 = conv.MaxPooling(2, 6)
        self.C3 = conv.Convolution_3D(6,5,16,self.lr)
        self.relu2=tool.ReLU()
        self.P4 = conv.MaxPooling(2, 16)
        self.C5 = network.Neuron(400,120,self.lr)
        self.relu3=tool.ReLU()
        self.F6 = network.Neuron(120, 84, self.lr)
        self.relu4=tool.ReLU()
        self.F7 = network.Neuron(84, 10, self.lr)
        self.O8 = tool.softmaxWithLoss()

        if(is_debug==False):
            self.batch_size=constant.batch_size_for_normal
        else:
            self.batch_size=constant.batch_size_for_debug
        self.train_accuracy_cnt=[]
        self.test_accuracy_cnt=[]
        self.train_loss=[]
        self.test_loss=[]

        self.batchs=None
        self.loss_sum_for_cur_batch=None
        self.accuracy_for_cur_batch=None

    def forward(self,data,label,is_test_phase=False):
        # forward starts
        I0_output = self.I0.forward(data)
        C1_input = self.C1.con_format_input(I0_output)
        C1_output = self.C1.forward(C1_input, self.I0.target_size, 1, 0)
        AF1_output = self.relu1.forward(C1_output)
        P2_output = self.P2.forward(AF1_output, self.C1.output_size)

        # for conv 3D
        C3_output = self.C3.forward(P2_output,14)
        AF2_output = self.relu2.forward(C3_output)
        P4_output = self.P4.forward(AF2_output, self.C3.output_size)

        C5_output = self.C5.forward(np.array(P4_output).reshape(1,-1)[0])
        AF3_output = self.relu3.forward(np.array(C5_output).reshape(-1,120))
        F6_output = self.F6.forward(AF3_output)
        AF4_output = self.relu4.forward(F6_output.reshape(1,-1))
        F7_output = self.F7.forward(AF4_output)[0]
        predict_label = np.argmax(F7_output)
        # get the predict output
        is_predict_rst_right = None
        if(np.argmax(label)==predict_label):
            is_predict_rst_right=1
        else:
            is_predict_rst_right=0
        # get loss
        loss=self.O8.forward(F7_output,label)

        return loss,is_predict_rst_right,predict_label

    def backward(self):
        dout_O8 = self.O8.backward()
        dout_F7 = self.F7.backward(dout_O8.reshape(1, -1))
        dout_AF4 = self.relu4.backward(dout_F7.reshape(1, -1))
        dout_F6 = self.F6.backward(dout_AF4)
        dout_AF3 = self.relu3.backward(dout_F6[0])

        # code for C5_FC
        ll=[]
        ll.append(dout_AF3)
        self.C5.x=self.C5.x.reshape(1,400)
        dout_C5 = self.C5.backward(dout_AF3)[0]
        dout_C5_format=[]
        for i in range(16):
            temp_dout_C5=np.array(dout_C5[i*25:i*25+25]).reshape(5,5)
            dout_C5_format.append(temp_dout_C5)
        dout_P4 = self.P4.backward(dout_C5_format)
        dout_AF2= self.relu2.backward(dout_P4)
        dout_C3 = self.C3.backward(dout_AF2)

        dout_P2 = self.P2.backward(dout_C3)
        dout_AF1 = self.relu1.backward(dout_P2)
        dout_C1 = self.C1.backward(dout_AF1)

    def update(self):
        self.C1.update()
        self.C3.update()
        self.C5.update()
        self.F6.update()
        self.F7.update()

    def train(self,epoch_index):
        self.loss_sum_for_cur_batch=0
        self.accuracy_for_cur_batch=0
        self.train_loss=[]
        self.train_accuracy_cnt=[]
        self.train_batchs=len(self.train_data)/self.batch_size
        fig,ax=plt.subplots()
        x_loss=[0,0]
        y_loss=[1,1]
        x_acc = [0, 0]
        y_acc = [1, 1]
        plt.title('train')
        plt.ion()
        for i in range(self.train_batchs):
            self.loss_sum_for_cur_batch=0.0
            self.accuracy_for_cur_batch=0.0
            print("train_bacth_num:%d start" % i)
            start_time=time.time()
            for j in range(self.batch_size):
                index_for_total=i*self.batch_size+j
                loss,accuracy,predict_label=self.forward(self.train_data[index_for_total],self.train_label[index_for_total])
                self.loss_sum_for_cur_batch += loss
                self.accuracy_for_cur_batch += accuracy
                self.backward()
                self.update()
                if(self.is_debug):
                    print("train_batch_num %d predict_label %d true_label %d" % (i, predict_label, np.argmax(self.train_label[index_for_total])))
            end_time=time.time()
            loss_cur_average=self.loss_sum_for_cur_batch / self.batch_size
            accuracy_for_average=(self.accuracy_for_cur_batch) / self.batch_size
            print("train_batch_num %d uses time %fs" % (i,end_time-start_time))
            print("train_batch_num:%d---loss:%f---accuracy:%f---" % (i,loss_cur_average,accuracy_for_average))
            print

            self.train_loss.append(loss_cur_average)
            self.train_accuracy_cnt.append(accuracy_for_average)
            if(i==0):
                x_loss[1]=i
                y_loss[1]=loss_cur_average
                x_acc[1]=i
                y_acc[1]=accuracy_for_average
                continue
            x_loss[0]=x_loss[1]
            y_loss[0]=y_loss[1]
            x_loss[1]=i
            y_loss[1]=loss_cur_average
            x_acc[0] = x_acc[1]
            y_acc[0] = y_acc[1]
            x_acc[1] = i
            y_acc[1] = accuracy_for_average
            # plt.plot(x_loss, x_loss)
            # plt.plot(x_acc, y_acc)
            plt.plot(x_loss, y_loss, 'b-', color='red', label='train_loss')
            plt.plot(x_acc, y_acc, 'b-', label='train_accuracy')
            plt.pause(0.01)
        plt.savefig("../rstImg/train"+str(epoch_index)+".png")

    def test(self,epoch_index):
        self.test_loss = []
        self.test_accuracy_cnt = []
        self.test_batchs = len(self.test_data) / self.batch_size
        fig,ax=plt.subplots()
        x_loss=[0,0]
        y_loss=[1,1]
        x_acc = [0, 0]
        y_acc = [1, 1]
        plt.title('test')
        plt.ion()
        for i in range(self.test_batchs):
            self.loss_sum_for_cur_batch = 0.0
            self.accuracy_for_cur_batch = 0.0
            print("test_bacth_num:%d start" % i)
            # self.init_batch(i)
            start_time=time.time()
            for j in range(self.batch_size):
                index_for_total = i * self.batch_size + j
                loss, accuracy,predict_label = self.forward(self.test_data[index_for_total], self.test_label[index_for_total])
                self.loss_sum_for_cur_batch += loss
                self.accuracy_for_cur_batch += accuracy
            end_time=time.time()
            loss_cur_average = self.loss_sum_for_cur_batch / self.batch_size
            accuracy_for_average = self.accuracy_for_cur_batch / self.batch_size
            print("test_batch_num %d uses time %fs" % (i,end_time-start_time))
            print("test_bacth_num:%d---loss:%f---accuracy:%f---" % (i, loss_cur_average, accuracy_for_average))
            print
            self.test_loss.append(loss_cur_average)
            self.test_accuracy_cnt.append(accuracy_for_average)

            if (i == 0):
                x_loss[1] = i
                y_loss[1] = loss_cur_average
                x_acc[1] = i
                y_acc[1] = accuracy_for_average
                continue
            x_loss[0] = x_loss[1]
            y_loss[0] = y_loss[1]
            x_loss[1] = i
            y_loss[1] = loss_cur_average
            x_acc[0] = x_acc[1]
            y_acc[0] = y_acc[1]
            x_acc[1] = i
            y_acc[1] = accuracy_for_average
            plt.plot(x_loss, y_loss, 'b-', color='red', label='train_loss')
            plt.plot(x_acc, y_acc, 'b-', label='train_accuracy')
            plt.pause(0.01)
        plt.savefig("../rstImg/test"+str(epoch_index)+".png")

    def draw_graph(self,is_draw_test=False):
        fig = plt.figure('fig')
        x_axis = range(0,self.test_batchs)
        plt.xlabel('batches')
        if(is_draw_test==False):
            plt.plot(x_axis, self.train_loss, 'b-', color='red',label='train_loss')
            plt.plot(x_axis, self.train_accuracy_cnt, 'b-', label='train_accuracy')
            plt.title('train')
            plt.ylabel('value')
        else:
            plt.plot(x_axis, self.test_loss, 'b-', color='red',label='test_loss')
            plt.plot(x_axis, self.test_accuracy_cnt, 'b-', label='test_accuracy')
            plt.title('test')
            plt.ylabel('value')
        plt.legend(loc='lower right')
        plt.show(fig)

    def export_weight(self):
        file = open('../weights' + tool.get_now_time() + '.txt', 'w')
        weights = {}
        weights['C1'] = [data.tolist() for data in self.C1.W]
        weights['C3'] = [data.tolist() for data in self.C3.W]
        weights['C5'] = {'W': self.C5.W.tolist(), 'b': self.C5.b.tolist()}
        weights['F6'] = {'W': self.F6.W.tolist(), 'b': self.F6.b.tolist()}
        weights['F7'] = {'W': self.F7.W.tolist(), 'b': self.F7.b.tolist()}
        weights_json = json.dumps(weights)
        file.write(weights_json)
        file.close()

    def load_weight(self, filename):
        file = open('../' + filename, 'r')
        line = file.readline()
        weights = json.loads(line)
        # print weights['C1']
        # print weights['C3']
        # print weights['C5']
        # print weights['F6']
        # print weights['F7']
        self.C1.W = [np.array(data) for data in weights['C1']]
        self.C3.W = [np.array(data) for data in weights['C3']]
        self.C5.W = [np.array(weights['C5']['W'])]
        self.C5.b = [np.array(weights['C5']['b'])]
        self.F6.W = [np.array(weights['F6']['W'])]
        self.F6.b = [np.array(weights['F6']['b'])]
        self.F7.W = [np.array(weights['F7']['W'])]
        self.F7.b = [np.array(weights['F7']['b'])]
        file.close()

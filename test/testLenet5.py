import lenet5
import conv
import numpy as np
from pympler import tracker
import time

def test_C3():
    C3 = lenet5.lenet5_C3_conv()
    for times in range(1):
        start_time = time.clock()
        ll = []
        for i in range(6):
            ll.append(np.random.rand(14, 14))
        dout = np.random.randn(16, 10, 10)
        forward_output=C3.forward(ll)
        print "forward"
        print np.array(forward_output).shape
        print np.array(forward_output)
        backward_output=C3.backward(dout)

        print np.array(backward_output)
        print np.array(backward_output).shape
        C3.update()
        end_time = time.clock()
        print("range %d uses %fs" %(times,(end_time-start_time)))
    return C3


def test_lenet5_class():
    lenet5_C=lenet5.lenet5_C()
    epoch=1
    for i in range(epoch):
        lenet5_C.train(i)
        lenet5_C.test(i)
        lenet5_C.export_weight()
        if (np.array(lenet5_C.test_accuracy_cnt).mean() > 0.8):
            lenet5_C.export_weight()


# C3=test_C3()
test_lenet5_class()
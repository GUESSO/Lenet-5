constantMin=1e-7
file_dir= '/home/lianchenyu/MNIST/MNIST_DATA/'
train_data_path=file_dir+'train-images.idx3-ubyte'
train_label_path=file_dir+'train-labels.idx1-ubyte'
test_data_path=file_dir+'t10k-images.idx3-ubyte'
test_label_path=file_dir+'t10k-labels.idx1-ubyte'

# nums of batch for debug
nums_for_debug=100
batch_size_for_normal=100
batch_size_for_debug=10

input_max=255

Weight_init_std=0.01
# Weight_init_std=1
lr=1e-3

lenet5_c3=[[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
          [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],
          [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
          [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],
          [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
          [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]]

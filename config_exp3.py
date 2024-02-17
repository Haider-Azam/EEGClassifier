# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications
data_folders = ['E:/exp3/crops/Normal_Crops','E:/exp3/crops/Abnormality_Crops']
processed_folder='E:/exp3/crops'
n_chans = 1
sampling_freq = 200
shuffle = True
model_name = 'shallow'#shallow/deep for DNN (deep terminal local 1)
n_start_chans = 25
n_chan_factor = 2  # relevant for deep model only
input_time_length = 600
stride=100  #stride is sampling frequency to ensure only one non-overlapping second for the windows
model_constraint = 'defaultnorm'
init_lr = 1e-3
batch_size = 64
max_epochs = 35 # until first stop, the continue train on train+valid
cuda = True # False

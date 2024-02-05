import numpy as np
import torch
import skorch
import h5py
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import elu,relu,leaky_relu
import braindecode 
from braindecode.models import *
from braindecode.models.modules import Expression
from braindecode.models.functions import squeeze_final_output,safe_log
from skorch.callbacks import Checkpoint
from skorch.helper import predefined_split
from config import *
from dataset import *
from sklearn.metrics import roc_auc_score
from mne import set_log_level
import h5py
from skorch.callbacks import LRScheduler
import os



#This dataset won't work when num_workers>0 on windows
class H5_Dataset(torch.utils.data.Dataset):
    def __init__(self, hd5_file,split):
        self.path = hd5_file
        super().__init__()
        self.split=split

    #def open_hdf5(self):
    #    self.h5_file = h5py.File(self.path, 'r')

    def __getitem__(self, index):
        #if not hasattr(self, 'h5_file'):
        #    self.open_hdf5()
        with h5py.File(self.path, 'r') as h5_file:
            window=h5_file[f'X_{self.split}'][index]
            label=h5_file[f'Y_{self.split}'][index]
        return window,label
 
    def __len__(self):
        #if not hasattr(self, 'h5_file'):
        #    self.open_hdf5()
        with h5py.File(self.path, 'r') as h5_file:
            return len(h5_file[f'Y_{self.split}'])
    

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, excel_path,input_time_length,stride):
        self.input_time_length=input_time_length
        self.stride=stride
        super().__init__()
        excel_file=pd.read_excel(excel_path)
        self.file_names=excel_file['name'].to_numpy(dtype=str)
        self.label=excel_file['label'].to_numpy()
        self.windows=excel_file['no_of_windows'].to_numpy()
        self.windows=np.cumsum(self.windows)


    def __getitem__(self, index):
        position=np.argmax(self.windows > index)
        #Calculates position relative to a recording
        if position>0:
            window_position=(index-self.windows[position-1])
        else:
            window_position=index
        window_position*=self.stride
        #print(self.file_names[position])
        with h5py.File(self.file_names[position], 'r') as h5_file:
            window=h5_file['x'][:,window_position : window_position + self.input_time_length]

        label=self.label[position]
        return window,label
 
    def __len__(self):
        return self.windows[-1]
    
    
if __name__=='__main__':
    set_log_level(False)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    device = 'cuda' if cuda else 'cpu'
    torch.backends.cudnn.benchmark = True
    #test_set=H5_Dataset('E:/exp1_1/test.hdf5','test')
    #train_set=H5_Dataset('E:/exp1_1/train.hdf5','train')
    test_set=WindowDataset(f'{processed_folder}/eval.xlsx',input_time_length,stride)
    train_set=WindowDataset(f'{processed_folder}/train.xlsx',input_time_length,stride)
    #Put model name here
    model_name="shallow"

    criterion=torch.nn.NLLLoss
    n_classes = 2
    if model_name=="shallow":
        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        #The final conv length is auto to ensure that output will give two values for single EEG window
        model = ShallowFBCSPNet(n_chans,
                                        n_classes,
                                        n_filters_time=n_start_chans,
                                        n_filters_spat=n_start_chans,
                                        n_times=input_time_length,
                                        final_conv_length='auto',)
        test=torch.ones(size=(7,21,6000))
        out=model.forward(test)
        print(out.shape)
    elif model_name == 'shallow_smac':
        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        #conv_nonlin = identity
        do_batch_norm = True
        drop_prob = 0.328794
        filter_time_length = 56
        n_filters_spat = 73
        n_filters_time = 24
        pool_mode = 'max'
        #pool_nonlin = identity
        pool_time_length = 84
        pool_time_stride = 3
        split_first_layer = True
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_filters_time,
                                n_filters_spat=n_filters_spat,
                                n_times=input_time_length,
                                final_conv_length='auto',
                                #conv_nonlin=conv_nonlin,
                                batch_norm=do_batch_norm,
                                drop_prob=drop_prob,
                                filter_time_length=filter_time_length,
                                pool_mode=pool_mode,
                                #pool_nonlin=pool_nonlin,
                                pool_time_length=pool_time_length,
                                pool_time_stride=pool_time_stride,
                                split_first_layer=split_first_layer,
                                )
        test=torch.ones(size=(7,21,6000))
        out=model.forward(test)
        print(out.shape)
    elif model_name=="deep":
        optimizer_lr = init_lr
        optimizer_weight_decay = 0
        model = Deep4Net(n_chans, n_classes,
                            n_filters_time=n_start_chans,
                            n_filters_spat=n_start_chans,
                            n_times=input_time_length,
                            n_filters_2 = int(n_start_chans * n_chan_factor),
                            n_filters_3 = int(n_start_chans * (n_chan_factor ** 2.0)),
                            n_filters_4 = int(n_start_chans * (n_chan_factor ** 3.0)),
                            final_conv_length='auto',
                            stride_before_pool=True)
        test=torch.ones(size=(6,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
    elif model_name=="deep_smac" or model_name == 'deep_smac_bnorm':
        optimizer_lr = 0.0000625
        if model_name == 'deep_smac':
                do_batch_norm = False
        else:
            do_batch_norm = True
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 14
        filter_length_4 = 32
        filter_time_length = 21
        #final_conv_length = 1
        first_nonlin = elu
        first_pool_mode = 'mean'
        later_nonlin = elu
        later_pool_mode = 'mean'
        n_filters_factor = 1.679066
        n_filters_start = 32
        pool_time_length = 1
        pool_time_stride = 2
        split_first_layer = True
        n_chan_factor = n_filters_factor
        n_start_chans = n_filters_start
        model = Deep4Net(n_chans, n_classes,
                n_filters_time=n_start_chans,
                n_filters_spat=n_start_chans,
                n_times=input_time_length,
                n_filters_2=int(n_start_chans * n_chan_factor),
                n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                final_conv_length='auto',
                stride_before_pool=True,
                drop_prob=drop_prob,
                filter_length_2=filter_length_2,
                filter_length_3=filter_length_3,
                filter_length_4=filter_length_4,
                filter_time_length=filter_time_length,
                first_conv_nonlin=first_nonlin,
                first_pool_mode=first_pool_mode,
                later_conv_nonlin=later_nonlin,
                later_pool_mode=later_pool_mode,
                pool_time_length=pool_time_length,
                pool_time_stride=pool_time_stride,
                split_first_layer=split_first_layer
                )
        test=torch.ones(size=(6,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
        del do_batch_norm,drop_prob,filter_length_2,filter_length_3,filter_length_4,filter_time_length,first_nonlin,n_chan_factor,n_start_chans,first_pool_mode,later_nonlin,later_pool_mode,n_filters_factor,n_filters_start,pool_time_length,pool_time_stride,split_first_layer
    #Works properly, fit the hybrid cnn
    elif model_name=="hybrid":
        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        #The final conv length is auto to ensure that output will give two values for single EEG window
        model = HybridNet(n_chans, n_classes,n_times=input_time_length,)
        test=torch.ones(size=(2,n_chans,input_time_length))
        out=model.forward(test)
        out_length=out.shape[2]
        model.final_layer=nn.Conv2d(100,n_classes,(out_length,1),bias=True,)
        model=nn.Sequential(model,nn.Flatten(),nn.LogSoftmax(dim=1))
        out=model.forward(test)
        print(out.shape)
        del out_length
    elif model_name=="TCN":
        criterion=torch.nn.NLLLoss
        import warnings
        #This disables the warning of the dropout2d layers receiving 3d input
        warnings.filterwarnings("ignore")
        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        n_blocks=7
        n_filters=32
        kernel_size=24
        drop_prob = 0.3
        x=TCN(n_chans,n_classes,n_blocks,n_filters,kernel_size,drop_prob)
        test=torch.ones(size=(7,n_chans,input_time_length))
        out=x.forward(test)
        out_length=out.shape[2]
        #There is no hyperparameter where output of TCN is (Batch_Size,Classes) when input is (Batch_Size,21,6000) so add new layers to meet size
        model=nn.Sequential(x,nn.Conv1d(n_classes,n_classes,out_length,bias=True,),nn.LogSoftmax(dim=1),nn.Flatten())
        out=model.forward(test)
        print(out.shape)
        del out_length,x
    elif model_name=="shallow_deep":
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 14
        filter_length_4 = 32
        n_filters_factor = 1.679066
        n_filters_start = 32
        split_first_layer = True
        n_chan_factor = n_filters_factor
        #n_start_chans = n_filters_start

        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        conv_time_length=25
        first_conv_nonlin=relu
        first_pool_nonlin=safe_log
        later_conv_nonlin=elu
        later_pool_nonlin=safe_log
        first_pool_mode = "mean"
        later_pool_mode = "mean"
        pool_time_length=15
        model = Deep4Net(n_chans, n_classes,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                n_times=input_time_length,
                                n_filters_2 = int(n_start_chans * n_chan_factor),
                                n_filters_3 = int(n_start_chans * (n_chan_factor ** 2.0)),
                                n_filters_4 = int(n_start_chans * (n_chan_factor ** 3.0)),
                                final_conv_length='auto',
                                first_pool_nonlin=first_pool_nonlin,
                                first_conv_nonlin=first_conv_nonlin,
                                #later_pool_nonlin=later_pool_nonlin,
                                #later_conv_nonlin=later_conv_nonlin,
                                filter_time_length=conv_time_length,
                                pool_time_length=pool_time_length,
                                first_pool_mode=first_pool_mode,
                                later_pool_mode=later_pool_mode,
                                split_first_layer=split_first_layer,
                                drop_prob=drop_prob,
                                filter_length_2=filter_length_2,
                                filter_length_3=filter_length_3,
                                filter_length_4=filter_length_4,
                                )
        test=torch.ones(size=(7,n_chans,input_time_length))
    #    out=model.forward(test)
    #    print(out.shape)

    elif model_name=="attention":
        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        model=ATCNet(n_chans,n_classes,input_time_length//sampling_freq,sampling_freq,concat=True)
        test=torch.ones(size=(7,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
    elif model_name=="transformer":
        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        #criterion=torch.nn.CrossEntropyLoss
        n_filters_time=20
        att_depth=4
        filter_time_length=15
        att_heads=3
        model=EEGConformer(n_outputs=n_classes,n_chans=n_chans,n_times=input_time_length,input_window_seconds=input_time_length//sampling_freq,
                        sfreq=sampling_freq,final_fc_length=7860,n_filters_time=n_filters_time,att_depth=att_depth,
                        filter_time_length=filter_time_length,att_heads=att_heads)
        test=torch.ones(size=(7,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
    if cuda:
        model.cuda()
    del test
    print(model_name)

    monitor = lambda net: any(net.history[-1, ('valid_accuracy_best','valid_f1_best','valid_loss_best')])
    cp=Checkpoint(monitor='valid_f1_best',dirname='model',f_params=f'{model_name}best_param.pkl',
               f_optimizer=f'{model_name}best_opt.pkl', f_history=f'{model_name}best_history.json')
    
    path=f'{model_name}II'
    classifier = braindecode.EEGClassifier(
            model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(test_set),
            optimizer__lr=optimizer_lr,
            #optimizer__weight_decay=optimizer_weight_decay,
            iterator_train=DataLoader,
            iterator_valid=DataLoader,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=2,
            iterator_valid__num_workers=2,
            iterator_train__persistent_workers=True,
            iterator_valid__persistent_workers=True,
            batch_size=batch_size,
            device=device,
            callbacks=["accuracy","f1",cp,skorch.callbacks.ProgressBar()],
            warm_start=True,
            )
    classifier.initialize()
    classifier.fit(train_set,epochs=5)
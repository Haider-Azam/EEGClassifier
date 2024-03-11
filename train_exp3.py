import numpy as np
import torch
import skorch
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
from config_exp3 import *
from dataset import *
from sklearn.metrics import roc_auc_score
from mne import set_log_level
from skorch.callbacks import LRScheduler
import os
ab_label_dict = ['Normal','Sharp Wave','Delta Slow Wave', 'Spike and Wave Discharge', 'Beta Wave', 'Theta Wave', 'Triphasic Wave', 'Low Voltage','Burst Suppression','Unknown']

class Exp3Dataset(torch.utils.data.Dataset):
    def __init__(self, excel_path,n_chans,input_time_length):
        super().__init__()
        excel_file=pd.read_excel(excel_path)
        self.file_names=excel_file['file_path'].to_numpy(dtype=str)
        self.label=excel_file['label'].to_numpy()
        self.ab_label=excel_file['ab_label'].to_numpy()
        self.n_chans=n_chans
        self.input_time_length=input_time_length
    def __getitem__(self, index):
        file=np.load(self.file_names[index])
        window=file['data']
        window=window.reshape(self.n_chans,self.input_time_length,order='F')
        #window=np.expand_dims(window,axis=-1)
        #print(window.shape)
        ab_label=np.array(self.ab_label[index])
        #ab_label=np.eye(len(ab_label_dict), dtype='uint8')[ab_label]
        return window,ab_label
 
    def __len__(self):
        return len(self.file_names)
    
    
if __name__=='__main__':
    device = 'cuda' if cuda else 'cpu'
    
    set_log_level(False)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    torch.backends.cudnn.benchmark = True
    #test_set=H5_Dataset('E:/exp1_1/test.hdf5','test')
    #train_set=H5_Dataset('E:/exp1_1/train.hdf5','train')
    
    #Put model name here
    model_name="transformer"

    criterion=torch.nn.NLLLoss
    n_classes = len(ab_label_dict)
    
    if model_name=="shallow":
        n_chans=2
        input_time_length=input_time_length//n_chans

        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        pool_time_length = 16
        pool_time_stride = 2
        filter_time_length = 80
        n_filters_spat = 64
        n_filters_time = 32
        split_first_layer = True
        drop_prob = 0.328794
        #The final conv length is auto to ensure that output will give two values for single EEG window
        model = ShallowFBCSPNet(n_chans,
                                        n_classes,
                                        #n_filters_time=n_start_chans,
                                        #n_filters_spat=n_start_chans,
                                        n_times=input_time_length,
                                        pool_time_length=pool_time_length,
                                        pool_time_stride=pool_time_stride,
                                        drop_prob=drop_prob,
                                        n_filters_time=n_filters_time,
                                        n_filters_spat=n_filters_spat,
                                        filter_time_length=filter_time_length,
                                        split_first_layer=split_first_layer,
                                        final_conv_length='auto',)
        test=torch.ones(size=(7,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
    elif model_name == 'shallow_smac':
        n_chans=2
        input_time_length=input_time_length//n_chans

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
        test=torch.ones(size=(7,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
    elif model_name=="deep":
        n_chans=2
        input_time_length=input_time_length//n_chans

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
        n_chans=2
        input_time_length=input_time_length//n_chans

        optimizer_lr = 0.0000625
        if model_name == 'deep_smac':
                do_batch_norm = False
        else:
            do_batch_norm = True
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 16
        filter_length_4 = 20
        filter_time_length = 8
        #final_conv_length = 1
        first_nonlin = elu
        first_pool_mode = 'mean'
        later_nonlin = elu
        later_pool_mode = 'mean'
        n_filters_factor = 1.679066
        n_filters_start = 32
        pool_time_length = 1
        pool_time_stride = 1
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
        #out=model.forward(test)
        #print(out.shape)
    elif model_name=="TCN":
        n_chans=1
        input_time_length=input_time_length//n_chans

        criterion=torch.nn.NLLLoss
        import warnings
        #This disables the warning of the dropout2d layers receiving 3d input
        warnings.filterwarnings("ignore")
        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        n_blocks=5
        n_filters=32
        kernel_size=5
        drop_prob = 0.4
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
        n_chans=2
        input_time_length=input_time_length//n_chans

        drop_prob = 0.244445
        filter_length_2 = 8
        filter_length_3 = 12
        filter_length_4 = 16
        n_filters_factor = 1.679066
        n_filters_start = 32
        split_first_layer = True
        n_chan_factor = n_filters_factor
        #n_start_chans = n_filters_start

        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        filter_time_length=4
        first_conv_nonlin=relu
        first_pool_nonlin=safe_log
        later_conv_nonlin=elu
        later_pool_nonlin=safe_log
        first_pool_mode = "mean"
        later_pool_mode = "mean"
        pool_time_length=2
        pool_time_stride=2
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
                            filter_time_length=filter_time_length,
                            pool_time_length=pool_time_length,
                            pool_time_stride=pool_time_stride,
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
        n_chans=1
        input_time_length=input_time_length//n_chans

        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        model=ATCNet(n_chans,n_classes,input_time_length//sampling_freq,sampling_freq,concat=True)
        test=torch.ones(size=(7,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
    elif model_name=="transformer":
        n_chans=1
        input_time_length=input_time_length//n_chans

        optimizer_lr = 0.0000625
        optimizer_weight_decay = 0
        #criterion=torch.nn.CrossEntropyLoss
        n_filters_time=32
        att_depth=3
        filter_time_length=32
        att_heads=8
        add_log_softmax=False
        criterion=torch.nn.CrossEntropyLoss
        model=EEGConformer(n_outputs=n_classes,n_chans=n_chans,n_times=input_time_length,input_window_seconds=input_time_length//sampling_freq,
                        sfreq=sampling_freq,final_fc_length="auto",n_filters_time=n_filters_time,att_depth=att_depth,
                        filter_time_length=filter_time_length,att_heads=att_heads,add_log_softmax=add_log_softmax)
        test=torch.ones(size=(7,n_chans,input_time_length))
        out=model.forward(test)
        print(out.shape)
    if cuda:
        model.cuda()
    del test
    
    test_set=Exp3Dataset(f'{processed_folder}/eval.xlsx',n_chans,input_time_length)
    train_set=Exp3Dataset(f'{processed_folder}/train.xlsx',n_chans,input_time_length)
    print('input shape: ',train_set.__getitem__(0)[0].shape)
    print(model_name)


    monitor = lambda net: any(net.history[-1, ('valid_accuracy_best','valid_f1_best','valid_loss_best')])
    cp=Checkpoint(monitor='valid_acc_best',dirname='model',f_params=f'{model_name}best_param.pkl',
               f_optimizer=f'{model_name}best_opt.pkl', f_history=f'{model_name}best_history.json')
    
    path=f'{model_name}II'
    classifier = braindecode.EEGClassifier(
            model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(test_set),
            optimizer__lr=0.0000025,
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
            callbacks=["accuracy",cp,skorch.callbacks.ProgressBar()],#,"f1"
            warm_start=True,
            )
    classifier.initialize()
    try:
        classifier.load_params(f_params=f'model/{model_name}best_param.pkl', f_history=f'model/{model_name}best_history.json')
        print("Loading Succeded")
    except:
        print("Loading failed")
    classifier.fit(train_set,epochs=100)
    classifier.save_params(f_params=f'model/{path}_param.pkl', f_history=f'model/{path}_history.json')
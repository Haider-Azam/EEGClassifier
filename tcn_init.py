n_classes=2
model_name='TCN'
class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return self.__class__.__name__ + "(expression=%s) " % expression_str
if model_name=="TCN":
    import warnings
    #This disables the warning of the dropout2d layers receiving 3d input
    warnings.filterwarnings("ignore")
    optimizer_lr = 0.000625
    optimizer_weight_decay = 0
    n_blocks=7
    n_filters=32
    kernel_size=24
    drop_prob = 0.3
    add_log_softmax=False
    #Minimum time length for TCN, found inside tcn.py
    min_len = 1
    for i in range(n_blocks):
        dilation = 2 ** i
        min_len += 2 * (kernel_size - 1) * dilation
    print(f"Minimum length :{min_len}")
    #Only setting n_classes to 1 so TCN output is (batch,1,2) so we can remove additional conv1d block.
    #This is only possible due to input_time_length=30,n_block=3 and kernel_size=3.
    x=TCN(n_chans,n_classes,n_blocks,n_filters,kernel_size,drop_prob,n_times=input_time_length)
    test=torch.ones(size=(7,n_chans,input_time_length))
    out=x.forward(test)
    out_length=out.shape[2]
    #model=nn.Sequential(x,Expression(torch.squeeze),nn.LogSoftmax(dim=1))
    #There is no hyperparameter where output of TCN is (Batch_Size,Classes) when input is (Batch_Size,21,6000) so add new layers to meet size
    model=nn.Sequential(x,nn.Conv1d(n_classes,n_classes,out_length,bias=True,),Expression(torch.squeeze),nn.LogSoftmax(dim=1))
    out=model.forward(test)
    print(out.shape)
    del out_length,x
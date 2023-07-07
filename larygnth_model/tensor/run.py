from LSTM_SA_Architecture import LSTM_SA_Architecture
from dataprovider import SequenceDataProvider
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# --- check if all input arguments are available
if __name__ == '__main__':
    
    # --- define some varibles
    xsz=(256, 256)      # xsize
    ysz=(256, 256)      # ysize
    ndim=3              # ndimension                
    nfil=64             # nfilters
    rnfil=None          # reduced_nfilters
    layer=5             # nlayers
    cl=4                # nclasses
    loss="softmax"      # loss_function
    opt="adam"          # optimizer
    clw=None            # class_weights 
    lrate=0.0001        # learning_rate
    drate=None          # decay_rate
    bnorm=True          # bn
    reg=None            # reg
    regsc=0.00001       # reg_scale
    imgstd=False        # image_std                
    cc=True             # crop_concat
    cnfilt=False        # constant_nfilters
    nm="LSTMfull_rgb_nat"  # name                      
    verb=False          # verbose
    md=30000            # model
    imgsuf="_rgb.png"   # image_suffix              
    labsuf="_mask.png"  # label_suffix
    bg=True             # background
    bs=10               # batch_size
    ep=40               # epochs
    seq_len=10          # sequence_length   
    run_mode=1          # 1: train, 0:test

    # define paths to the dataset
    PATH_tain="./dataset/train/*.png"
    PATH_valid="./dataset/val/*.png"
    PATH_test="./dataset/test/*.png"

     # PLEASE DO NOT CHANGE!
    if clw == None:  # class_weights          
        clw_str = "None"
    else:
        clw = float(clw)
        clw_str = str(round(clw, 1))

    if drate == None:  # decay_rate
        drate_str = "None"
    else:
        drate = float(drate)
        drate_str = str(round(drate, 1))

    if reg == None:  # reg
        regu_str = "None"
    else:
        regu_str = None
    
    if bnorm == True:  # bnorm          
        bnorm_str = "True"
    elif bnorm == False:
        bnorm_str = "False"

    if bg == True:  # background          
        bg_str = "True"
    elif bg == False:
        bg_str = "False"


    # DataProvider doesn't count background as class
    ncl_data=int(cl)-1      

    
    if int(run_mode) == 1: #training
        
        
        # load data
        data_train = SequenceDataProvider(PATH_tain, sequence_length=int(seq_len), 
                                          image_suffix=str(imgsuf), label_suffix=str(labsuf), 
                                          nclasses=int(ncl_data), background=bg)
        data_valid = SequenceDataProvider(PATH_valid, sequence_length=int(seq_len), 
                                          image_suffix=str(imgsuf), label_suffix=str(labsuf), 
                                          nclasses=int(ncl_data), background=bg)

        # define NN-architecture
        lstmnet = LSTM_SA_Architecture(nfilters=int(nfil), nlayers=int(layer), 
                                       ndimensions=int(ndim), nclasses=int(cl), 
                                       class_weights=clw, loss_function=str(loss), 
                                       learning_rate=float(lrate), 
                                       nsequences=int(seq_len), decay_rate=None, bn=bnorm, 
                                       reg=None, reg_scale=float(regsc), image_std=str(imgstd), 
                                       constant_nfilters=str(cnfilt), name=str(nm))
        
        # compile the model
        lstmnet.compile(loss=str(loss), optimizer=tf.keras.optimizers.Adam(learning_rate=float(lrate)))
        
        # train NN    
        history = lstmnet.fit(x=data_train, validation_data=data_valid, epochs=int(ep))
        

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:45:38 2019

@author: joelrv
"""

def model_fn(train_data, lr, hidden_size, activation, aetype, batchnorm,
             dropout, input_dropout, ridge, l1_enc_coef):
    
    print("Backend is " + K.backend())
    print(" MB size of train_data" + str(getsizeof(train_data)/1000000))
    print(" Tuple size of adata" + str(adata.shape))
    if K.backend() == 'tensorflow':
      K.clear_session()
    gc.collect()
    net = AE_types[aetype](train_data[1].shape[1],
            hidden_size=hidden_size,
            l2_coef=0.0,
            l1_coef=0.0,
            l2_enc_coef=0.0,
            l1_enc_coef=l1_enc_coef,
            ridge=ridge,
            hidden_dropout=dropout,
            input_dropout=input_dropout,
            batchnorm=batchnorm,
            activation=activation,
            init='glorot_uniform',
            debug=args.debug)
    net.build()
    net.model.summary()

    optimizer = opt.__dict__['rmsprop'](lr=lr, clipvalue=5.0)
    net.model.compile(loss=net.loss, optimizer=optimizer)
    
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    return net.model
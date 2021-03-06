import os
import pickle
import json
from sys import getsizeof

import numpy as np
from kopt import CompileFN, test_fn
from hyperopt import fmin, tpe, hp, Trials, space_eval
import keras.optimizers as opt
from keras import backend as K

from . import io
from .network import AE_types
import gc

from . import data
from . import model

import linecache
import os
import tracemalloc


#from .base_anndata import AnnData

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

#tracemalloc.start()







def hyper(args):
    
    adata = io.read_dataset(args.input,
                        transpose=(not args.transpose),
                        test_split=False)

#    adata.write(os.path.join(args.outputdir, 'anndatabckup.h5ad'))
#    adata.write(os.path.join(args.outputdir, 'anndatabckup.h5ad'))
#    
#    del adata
#    
#    
##    adata = io.read_dataset(os.path.join(args.outputdir, 'anndatabckup.h5ad'),
##                            transpose=False,
##                            test_split=False)
#    adata = os.path.join(args.outputdir, 'anndatabckup.h5ad')
#    adata.isbacked=True                      
#    adata = io.normalize(adata,
#                         size_factors=args.sizefactors,
#                         logtrans_input=args.loginput,
#                         normalize_input=args.norminput)

    hyper_params = {
            "data": {
#                "inputData": hp.choice('d_input', (adata, adata)),
                #"inTranspose": hp.choice('d_inTranspose', (args.transpose, args.transpose)),
                "norm_input_log": hp.choice('d_norm_log', (True, False)),
                "norm_input_zeromean": hp.choice('d_norm_zeromean', (True, False)),
                "norm_input_sf": hp.choice('d_norm_sf', (True, False)),
                },
            "model": {
                "lr": hp.loguniform("m_lr", np.log(1e-3), np.log(1e-2)),
                "ridge": hp.loguniform("m_ridge", np.log(1e-7), np.log(1e-1)),
                "l1_enc_coef": hp.loguniform("m_l1_enc_coef", np.log(1e-7), np.log(1e-1)),
                "hidden_size": hp.choice("m_hiddensize", ((64,32,64), (32,16,32),
                                                          (64,64), (32,32), (16,16),
                                                          (16,), (32,), (64,), (128,))),
                "activation": hp.choice("m_activation", ('relu', 'selu', 'elu',
                                                         'PReLU', 'linear', 'LeakyReLU')),
                "aetype": hp.choice("m_aetype", ('zinb', 'zinb-conddisp')),
                "batchnorm": hp.choice("m_batchnorm", (True, False)),
                "dropout": hp.uniform("m_do", 0, 0.7),
                "input_dropout": hp.uniform("m_input_do", 0, 0.8),
                },
            "fit": {
                "epochs": args.hyperepoch
                }
    }

    def data_fn(norm_input_log, norm_input_zeromean, norm_input_sf):
        
        ad = adata.copy()
        ad = io.normalize(ad,
                     size_factors=norm_input_sf,
                     logtrans_input=norm_input_log,
                     normalize_input=norm_input_zeromean)
                     
        x_train = {'count': ad.X, 'size_factors': ad.obs.size_factors}
#        print(x_train)
        #x_train = ad.X
        y_train = adata.X
#        print(y_train)
        gc.collect()
        return (x_train, y_train),

#    def model_fn(train_data, lr, hidden_size, activation, aetype, batchnorm,
#                 dropout, input_dropout, ridge, l1_enc_coef):
#        
#        print("Backend is " + K.backend())
#        print(" MB size of train_data" + str(getsizeof(train_data)/1000000))
##        if K.backend() == 'tensorflow':
##          K.clear_session()
#        gc.collect()
#        print(train_data[1].shape[1])
#        net = AE_types[aetype](train_data[1].shape[1],
#                hidden_size=hidden_size,
#                l2_coef=0.0,
#                l1_coef=0.0,
#                l2_enc_coef=0.0,
#                l1_enc_coef=l1_enc_coef,
#                ridge=ridge,
#                hidden_dropout=dropout,
#                input_dropout=input_dropout,
#                batchnorm=batchnorm,
#                activation=activation,
#                init='glorot_uniform',
#                debug=args.debug)
#        net.build()
#        net.model.summary()
#
#        optimizer = opt.__dict__['rmsprop'](lr=lr, clipvalue=5.0)
#        net.model.compile(loss=net.loss, optimizer=optimizer)
#        
#        snapshot = tracemalloc.take_snapshot()
#        display_top(snapshot)
#
#        return net.model

    output_dir = os.path.join(args.outputdir, 'hyperopt_results')
    objective = CompileFN('autoencoder_hyperpar_db', 'myexp1',
                          data_fn=data_fn,
                          model_fn=model.model_fn,
                          loss_metric='loss',
                          loss_metric_mode='min',
                          valid_split=.2,
                          save_model=None,
                          save_results=True,
                          use_tensorboard=False,
                          save_dir=output_dir)

    test_fn(objective, hyper_params, save_model=None)

    trials = Trials()
    best = fmin(objective,
                hyper_params,
                trials=trials,
                algo=tpe.suggest,
                max_evals=args.hypern,
                catch_eval_exceptions=True)

    with open(os.path.join(output_dir, 'trials.pickle'), 'wb') as f:
        pickle.dump(trials, f)

    #TODO: map indices in "best" back to choice-based hyperpars before saving
    with open(os.path.join(output_dir, 'best.json'), 'wt') as f:
        json.dump(best, f, sort_keys=True, indent=4)
        json.dump(space_eval(hyper_params, trials.argmin), f, sort_keys=True, indent=4)

    print(best)
    print(space_eval(hyper_params, trials.argmin))
    #TODO: not just save the best conf but also train the model with these params

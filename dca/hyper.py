import os
import pickle
import json
from sys import getsizeof

import numpy as np
from kopt import CompileFN, test_fn
from hyperopt import fmin, tpe, hp, Trials
import keras.optimizers as opt
from keras import backend as K

from . import io
from .network import AE_types
import gc


import linecache
import os
import tracemalloc

from . import data
from . import model

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

tracemalloc.start()



def hyper(args):
    adata = io.read_dataset(args.input,
                            transpose=args.transpose,
                            test_split=False)
                            
#    adata = io.normalize(adata,
#                         size_factors=args.sizefactors,
#                         logtrans_input=args.loginput,
#                         normalize_input=args.norminput)

    hyper_params = {
            "data": {
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


    output_dir = os.path.join(args.outputdir, 'hyperopt_results')
    objective = CompileFN('autoencoder_hyperpar_db', 'myexp1',
                          data_fn=data.data_fn(adata),
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

    print(best)

    #TODO: not just save the best conf but also train the model with these params

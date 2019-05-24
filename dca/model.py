# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:49:40 2019

@author: joelrv
"""

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

def model_fn(train_data, lr, hidden_size, activation, aetype, batchnorm,
             dropout, input_dropout, ridge, l1_enc_coef):
    
    print("Backend is " + K.backend())
    print(" MB size of train_data" + str(getsizeof(train_data)/1000000))
#        if K.backend() == 'tensorflow':
#          K.clear_session()
    gc.collect()
    print(train_data[1].shape[1])
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
            debug=True)
    net.build()
    net.model.summary()

    optimizer = opt.__dict__['rmsprop'](lr=lr, clipvalue=5.0)
    net.model.compile(loss=net.loss, optimizer=optimizer)
    
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    return net.model
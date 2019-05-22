# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:42:28 2019

@author: joelrv
"""
from . import io

def data_fn(adata,norm_input_log, norm_input_zeromean, norm_input_sf):

    ad = adata.copy()
    ad = io.normalize(ad,
                      size_factors=norm_input_sf,
                      logtrans_input=norm_input_log,
                      normalize_input=norm_input_zeromean)


    x_train = {'count': ad.X, 'size_factors': ad.obs.size_factors}
    #x_train = ad.X
    y_train = ad.raw.X
    return (x_train, y_train),

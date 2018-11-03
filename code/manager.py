# -*- coding: utf-8 -*-

import sys
from CapsuleNN.train_capsule import main_train as cb
from BiGRU.train_bigru import BiGRU

if __name__ == '__main__':
    model_index = sys.argv[1]
    batch_type = sys.argv[2]
    if batch_type == 'capsule':
        cb(model_index)
    elif batch_type == 'bigru':
        bigru = BiGRU(maxlen=1200)
        bigru.train(model_index=model_index)
    else:
        print('batch_type is error')

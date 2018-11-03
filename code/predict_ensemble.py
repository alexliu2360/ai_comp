# -*- coding: utf-8 -*-

from CapsuleNN.predict_capsule import part_capsule_predict
from BiGRU.predict_bigru import part_bigru_predict
from config import predict_ensemble_fn, predict_ensemble_prob_fn

if __name__ == '__main__':
    # part_capsule_predict([3, 6, 8, 9, 14, 15, 16, 19, 20], predict_fn=predict_ensemble_fn, predict_prob_fn=predict_ensemble_prob_fn)
    # part_bigru_predict([1, 2, 4, 5, 7, 10, 11, 12, 13, 17, 18], predict_fn=predict_ensemble_fn,
    #                    predict_prob_fn=predict_ensemble_prob_fn, mode='a')
    part_bigru_predict([18], predict_fn=predict_ensemble_fn,
                       predict_prob_fn=predict_ensemble_prob_fn, mode='w')

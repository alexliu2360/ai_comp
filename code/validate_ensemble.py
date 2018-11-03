# -*- coding: utf-8 -*-

from CapsuleNN.validation_capsule import part_capsule_validate
from BiGRU.validation_bigru import part_bigru_validate
from config import validate_ensemble_fn, validate_ensemble_prob_fn

if __name__ == '__main__':
    part_capsule_validate([3, 6, 8, 9, 14, 15, 16, 19, 20], validate_fn=validate_ensemble_fn,
                          validate_prob_fn=validate_ensemble_prob_fn)
    part_bigru_validate([1, 2, 4, 5, 7, 10, 11, 12, 13, 17, 18], validate_fn=validate_ensemble_fn,
                        validate_prob_fn=validate_ensemble_prob_fn)

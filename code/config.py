# -*- coding: utf-8 -*-
import os

config_path = os.path.abspath(os.path.dirname(__file__)) + '/'
print(config_path)  # Fine-grained_Sentiment_Analysis/code/AI-Comp-master/code
stopwords_path = config_path
stopwords_fn = 'stopwords.txt'

original_data_path = config_path + '../data/original/'
original_train_fn = 'sentiment_analysis_trainingset.csv'
original_validate_fn = 'sentiment_analysis_validationset.csv'
original_test_fn = 'sentiment_analysis_testa.csv'

validate_data_path = config_path + '../data/validate/'
validate_bigru_fn = 'validate_bigru.csv'
validate_bigru_prob_fn = 'validate_bigru_prob.csv'
validate_capsule_fn = 'validate_capsule.csv'
validate_capsule_prob_fn = 'validate_capsule_prob.csv'
validate_ensemble_fn = 'validate_ensemble.csv'
validate_ensemble_prob_fn = 'validate_ensemble_prob.csv'

predict_data_path = config_path + '../data/predict/'
predict_bigru_fn = 'predict_bigru.csv'
predict_bigru_prob_fn = 'predict_bigru_prob.csv'
predict_capsule_fn = 'predict_capsule.csv'
predict_capsule_prob_fn = 'predict_capsule_prob.csv'
predict_ensemble_fn = 'predict_ensemble.csv'
predict_ensemble_prob_fn = 'predict_ensemble_prob.csv'


preprocess_data_path = config_path + '../data/preprocess/'
preprocess_train_data_fn = "train_char.csv"
preprocess_validate_data_fn = "validation_char.csv"
preprocess_testa_data_fn = "test_char.csv"

test_train_data_file = config_path + '../data/test/test_train_10.csv'
test_validate_data_file = config_path + '../data/test/test_validate_10.csv'


capslue_models_path = config_path + "models/capslue/"
bigru_models_path = config_path + "models/bigru/"
rcnn_models_path = config_path + "models/rcnn/"

bigru_embeddings_matrix_path = config_path+'BiGRU/'
bigru_embeddings_matrix_fn = 'bigru_embeddings_matrix'
bigru_embeddings_matrix_suffix = 'npy'

word2vec_chars_path = config_path + "../data/word2vec_chars/"
word2vec_chars_fn = "chars.vector"

tokenizer_bigru_path = config_path + 'BiGRU/'
tokenizer_bigru_fn = 'tokenizer_char.pickle'

tokenizer_capsule_path = config_path + 'CapsuleNN/'
tokenizer_capsule_fn = 'tokenizer_char.pickle'

column_batch_map = {1: 'ltc', 2: 'ldfbd',
                    3: 'letf', 4: 'swt',
                    5: 'swa', 6: 'spc',
                    7: 'ssp', 8: 'pl',
                    9: 'pce', 10: 'pd',
                    11: 'ed', 12: 'en',
                    13: 'es', 14: 'ec',
                    15: 'dp', 16: 'dt',
                    17: 'dl', 18: 'dr',
                    19: 'ooe', 20: 'owta'}

column_list = ["",  # 加一个空值，这样可以从1开始计数
               "location_traffic_convenience", "location_distance_from_business_district",
               "location_easy_to_find", "service_wait_time",
               "service_waiters_attitude", "service_parking_convenience",
               "service_serving_speed", "price_level",
               "price_cost_effective", "price_discount",
               "environment_decoration", "environment_noise",
               "environment_space", "environment_cleaness",
               "dish_portion", "dish_taste",
               "dish_look", "dish_recommendation",
               "others_overall_experience", "others_willing_to_consume_again"
               ]

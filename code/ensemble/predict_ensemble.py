import pandas as pd


def get_classification(arr1, arr2):
    arr1 = arr1.str.lstrip("[")
    arr1 = arr1.str.rstrip("]")
    arr1 = arr1.str.split(" ")
    arr2 = arr2.str.lstrip("[")
    arr2 = arr2.str.rstrip("]")
    arr2 = arr2.str.split(" ")
    label = pd.Series([0] * len(arr1))
    for ind in arr1.index:
        label_tmp1 = [float(val) for val in arr1[ind] if val]
        label_tmp2 = [float(val) for val in arr2[ind] if val]
        label_tmp = list()
        for i, ele in enumerate(label_tmp1):
            label_tmp.append(0.5 * label_tmp1[i] + 0.5 * label_tmp2[i])
        if label_tmp.index(max(label_tmp)) == 0:
            label[ind] = -2
        elif label_tmp.index(max(label_tmp)) == 1:
            label[ind] = -1
        elif label_tmp.index(max(label_tmp)) == 2:
            label[ind] = 0
        else:
            label[ind] = 1
    return label


predict_bigru_char = pd.read_csv("best_model_1/submit/baseline_bigru_char_prob.csv")
predict_capsule_char = pd.read_csv("best_model_1/submit/baseline_capsule_char_prob.csv")
predict_submit_char = predict_capsule_char.copy()
predict_submit_char.iloc[:, 2:] = predict_bigru_char.iloc[:, 2:] + predict_capsule_char.iloc[:, 2:]
for col in predict_submit_char.columns[2:]:
    predict_submit_char[col] = get_classification(predict_bigru_char[col], predict_capsule_char[col])
print(predict_submit_char)
predict_submit_char.to_csv("best_model_1/submit/baseline_ensemble_char.csv", index=False)

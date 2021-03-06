import numpy as np

from logistic_regression_train import sig

def load_weight(w):

    f = open(w)

    w = []

    for line in f.readlines():
        lines = line.strip().split("\t")

        w_tmp = []

        for x in lines:
            w_tmp.append(float(x))

        w.append(w_tmp)

    f.close()

    return np.mat(w)

def load_data(file_name, n):

    f = open(file_name)

    feature_data = []

    for line in f.readlines():

        feature_tmp = []

        lines = line.strip().split("\t")

        if len(lines) != n - 1:

            continue

        feature_tmp.append(1)

        for x in lines:

            feature_tmp.append(float(x))

        feature_data.append(feature_tmp)

    f.close()

    return np.mat(feature_data)


def predict(data, w):

    h = sig(data * w.T)

    m = np.shape(h)[0]

    for i in range(m):

        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0

    return h

def save_result(file_name, result):

    m = np.shape(result)[0]

    tmp = []

    for i in range(m):
        tmp.append(str(result[i, 0]))

        f_result = open(file_name, "w")

        f_result.write("\t".join(tmp))

        f_result.close()



if __name__ == "__main__":

    print("Load model")

    w = load_weight("weights")
    n = np.shape(w)[1]

    print("Load data")

    testData = load_data("test_data", n)

    print("Get prediction")

    h = predict(testData, w)

    print("Save prediction")

    save_result("result", h)

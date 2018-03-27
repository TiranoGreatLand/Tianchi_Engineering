import numpy as np
import pandas as pd
import pickle
import time
from sklearn.decomposition import PCA
import tensorflow as tf

np.random.seed(int(time.time()))

def dict2list(dic):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

# input one array and return the most frequent one
def MostFrequentOne(column):
    itemfreq = {}
    for i in column:
        if i not in itemfreq:
            itemfreq[i] = 1
        else:
            itemfreq[i] += 1
    tmp_dict = sorted(dict2list(itemfreq), key=lambda d:d[1], reverse=True)
    return tmp_dict[0][0]

# return a reverse boolean column
def BoolReverse(column):
    x = np.zeros(len(column))
    x[column] = 1
    y = (x==0)
    return y

#If A column with only one value equals to zero, then it shall be excluded
def OneValueIfNeo(column):
    x = np.sum(column==0)
    if x > 0:
        return True
    else:
        return False

# normalize raw, no mean and std, they shall be computed
def Normalize_Raw(column):
    mean = np.mean(column)
    std = np.std(column)
    return mean, std
# normalize one column by mean and std
def Norm(column, mean, std):
    return (column - mean)/std

# 0-1 average
def NeoOneMean_Raw(column):
    maxv = np.max(column)
    minv = np.min(column)
    return maxv, minv
# mean one column to 0-1
def NeoOneMean(column, maxv, minv):
    return (column - minv) / (maxv - minv)

# fix the non or null
def OneStrColumnFix(column, nn):
    mfo = MostFrequentOne(column[nn])
    rn = BoolReverse(nn)
    if np.sum(rn) > 0:
        column[rn] = mfo
    return column

# one_hot to map str to int
def OneHot_Str(column, retdict = 0, s2nd = None):
    cl = len(column)
    if s2nd is None:
        syms = set(column)
        sim = {}
        count = 0
        for s in syms:
            sim[s] = count
            count += 1
        retoh = np.zeros((cl, count))
    else:
        sim = s2nd
        retoh = np.zeros((cl, len(s2nd)))
    for i in range(cl):
        s = column[i]
        mi = sim[s]
        retoh[i, mi] = 1.0
    if retdict == 0:
        return retoh
    else:
        return retoh, sim

def LineaModel(input_data):
    l1 = tf.layers.dense(input_data, 1000)
    l2 = tf.layers.dense(l1, 100)
    l3 = tf.layers.dense(l2, 10)
    l4 = tf.layers.dense(l3, 1)
    return l4

def MSE(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred-y))

def Loss_F(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

class DataProcessing(object):
    def __init__(self, test_path='测试A.xlsx'):
        #read the data
        self.id_num = pickle.load(open('id_num.txt', 'rb'))
        self.train_X = pickle.load(open('train_data.txt', 'rb'))
        self.train_y = pickle.load(open('train_y.txt', 'rb'))
        self.non_null = pickle.load(open('non_null.txt', 'rb'))
        print("shape of train", self.train_X.shape, self.non_null.shape)
        self.firstdelete = pickle.load(open('firstdelete.txt', 'rb'))
        self.num_column = self.train_X.shape[1]
        self.test_data = pd.read_excel(test_path)
        print("read necessary data over")
        self.types = self.GetColumnTypes(self.train_X, self.non_null)

        self.str2num = None
        self.mfs = None
        self.methods = None
        self.pca = None
        self.y_m = None
        self.y_s = None
        self.x_pca_norm = None
        self.test_id = None
        self.test_X = None

        self.DataMerge()
        self.train_X = self.DataFix(self.train_X, self.non_null)
        self.DataNormalization()
        self.TrainTestSplit()

        self.input_X = None
        self.input_y = None
        self.pred_y = None
        self.loss = None
        self.opt = None
        self.sess = None

        print("modeling")
        self.Modeling()
        self.ModelInit()
        self.Train()
        self.AnsWriter()
        print("Answer has been made")
        self.ModelClose()
        print("close and over")

    def TrainTestSplit(self):
        train_X = self.train_X[:, :500]
        test_X = self.train_X[:, 500:]
        self.train_X = train_X
        self.test_X = test_X
        print("data splited")


    def DataMerge(self):
        test_data = self.test_data.values
        non_null = self.test_data.isnull().values
        test_data = np.delete(test_data, self.firstdelete, axis=1)
        non_null = np.delete(non_null, self.firstdelete, axis=1)
        non_null = non_null[:, 1:]
        test_X = test_data[:, 1:]
        self.test_id = test_data[:, 0]
        assert test_X.shape == non_null.shape
        print(self.train_X.shape, test_X.shape)
        self.train_X = np.concatenate((self.train_X, test_X), axis=0)
        self.non_null = np.concatenate((self.non_null, non_null), axis=0)
        print("data merged")

    def GetColumnTypes(self, columns, non_null):
        types = []
        for i in range(self.num_column):
            data = columns[:, i]
            real = non_null[:, i]
            rd = data[real]
            x = str(type(rd[0]))
            if 'str' in x:
                types.append('str')
            elif 'float' in x:
                types.append('float')
            else:
                types.append('int')
        self.types = types
        print("types get over")
        return types

    def DataFix(self, columns, non_null, method=2):
        stridxs = []
        str2num = {}
        mfs = {}
        for i in range(self.num_column):
            t = self.types[i]
            if t == 'str':
                stridxs.append(i)
                column = columns[:, i]
                nn = non_null[:, i]
                rn = BoolReverse(nn)
                mc = MostFrequentOne(column[nn])
                mfs[i] = mc
                print(mc)
                if np.sum(rn) > 0:
                    column[rn] = mc
                    columns[:, i] = column
                one_hots, sim = OneHot_Str(column, retdict=1)
                str2num[i] = sim
                columns = np.concatenate((columns, one_hots), axis=1)
        print("str fixed, now fix null and delete str columns")
        self.str2num = str2num
        self.mfs = mfs
        methods = []
        for i in range(self.num_column):
            t = self.types[i]
            if t == 'str':
                methods.append((i, 0))
            else:
                data = columns[:, i]
                real = non_null[:, i]
                numv = len(set(data[real]))
                # fix null
                if np.sum(real) < 500:
                    rn = BoolReverse(real)
                    if numv == 1:
                        data[rn] = data[real][0]
                    else:
                        fixm = np.mean(data[real])
                        data[rn] = fixm
                        self.mfs[i] = fixm
                # fix null, then manipulate the column
                else:
                    self.mfs[i] = np.mean(data)
                if numv == 1:
                    data = np.ones(columns.shape[0])
                    methods.append((i, 1))
                else:
                    if method == 2:
                        mean, std = Normalize_Raw(data)
                        data = (data - mean) / std
                        methods.append((i, 2, mean, std))
                    elif method == 3:
                        maxv, minv = NeoOneMean_Raw(data)
                        data = (data - minv) / (maxv - minv)
                        methods.append((i, 3, maxv, minv))
                columns[:, i] = data
        print("data fixed")
        columns = np.delete(columns, stridxs, axis=1)
        print("now data fix over")
        self.methods = methods
        return columns

    def DataNormalization(self, components=15):
        pca = PCA(n_components=components)
        pca.fit(self.train_X)
        self.pca = pca
        self.train_X = pca.transform(self.train_X)
        self.num_column = self.train_X.shape[1]
        x_pca_norm = []
        for i in range(self.num_column):
            data = self.train_X[:, i]
            mean, std = Normalize_Raw(data)
            x_pca_norm.append((mean, std))
            data = (data - mean) / std
            self.train_X[:, i] = data
        self.x_pca_norm = x_pca_norm
        y_m = np.mean(self.train_y)
        y_s = np.std(self.train_y)
        self.train_y = (self.train_y-y_m)/y_s
        self.y_m = y_m
        self.y_s = y_s
        print("data normalized")


    def Modeling(self):
        self.input_X = tf.placeholder(shape=[None, self.num_column], dtype=tf.float32)
        self.input_y = tf.placeholder(shape=[None], dtype=tf.float32)
        self.pred_y = LineaModel(self.input_X)*self.y_s + self.y_m
        self.loss = Loss_F(self.pred_y, self.input_y*self.y_s+self.y_m)
        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def ModelInit(self):
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
    def Train(self, epoches=101):
        for i in range(epoches):
            _, l = self.sess.run([self.opt, self.loss], feed_dict={self.input_X:self.train_X, self.input_y:self.train_y})
            print(i, l)
    def ModelClose(self):
        self.sess.close()
    def TestPredict(self):
        test_pred = self.sess.run(self.pred_y, feed_dict={self.input_X:self.test_X})
        return test_pred

    def AnsWriter(self):
        test_pred = self.TestPredict()
        save = pd.DataFrame({'ID':self.test_id, 'value':test_pred})
        save.to_csv('answer.csv', index=False, header=False)

dp = DataProcessing()

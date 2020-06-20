import json
import numpy as np
import pandas as pd
import pickle
import re
import itertools
from collections import Counter
from collections import defaultdict
import nltk
from conf import conf
from nltk.corpus import stopwords
import tensorflow as tf
#import csv
import os
from nltk.tokenize import  word_tokenize
class dataloader():
    def __init__(self,path,file_name,args):
        '''

        :param path:
        :param file_name:
        :param args:
        '''
        self.TPS_DIR = path
        self.TP_file = os.path.join(self.TPS_DIR, file_name)
        self.args = args
        self.args["data"] = os.path.splitext(file_name)[0]

    def get_count(self,tp, id):  #获得id对应的数量 groupby函数
        playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
        count = playcount_groupbyid.size()
        return count

    def numerize(self,tp):
        user_id = [self.user2id[x] for x in tp['user_id']]
        item_id = [self.item2id[x] for x in tp['item_id']]
        tp['user_id'] = user_id
        tp['item_id'] = item_id
        return tp

    def load_file(self):
        f = open(self.TP_file)
        #保存评论的相关数据
        self.users_id = []
        self.items_id = []
        self.ratings = []
        self.reviews = []
        np.random.seed(2019)
        print("read file")
        for line in f: #过滤掉ID不清楚的评论
            js = json.loads(line)
            if str(js["reviewerID"])=="unknown":
                print("unknown")
                continue
            if str(js["asin"])=="unknown":
                print(("unknown2"))
                continue
            self.reviews.append(js['reviewText'])
            #print(js['reviewText'])
            self.users_id.append(str(js['reviewerID']))
            self.items_id.append(str(js['asin']))
            self.ratings.append(str(js['overall']))
        print(self.reviews[0])
        data = pd.DataFrame({'user_id': pd.Series(self.users_id),
                             'item_id': pd.Series(self.items_id),
                             'ratings': pd.Series(self.ratings),
                             'reviews': pd.Series(self.reviews)})[['user_id', 'item_id', 'ratings', 'reviews']]
        usercount, itemcount = self.get_count(data, 'user_id'), self.get_count(data, 'item_id')
        print("first itemcount",itemcount.shape[0])
        print("usercount:", usercount)
        print("itemcount:", itemcount)
        unique_uid = usercount.index  #unique_user_id
        unique_sid = itemcount.index
        self.item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))  # item id（标号）对应普通id
        self.user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))  # user_id（标号）对应普通id
        pickle.dump(self.item2id, open(os.path.join(self.TPS_DIR,self.args["data"]+'_item2id'), "wb+"))
        pickle.dump(self.user2id, open(os.path.join(self.TPS_DIR,self.args["data"]+'_user2id'), "wb+"))
        data = self.numerize(data)  # 处理好的 data(user_id,item_id,ratings,review)
        tp_rating = data[['user_id', 'item_id', 'ratings']]

        user_groupby = data.groupby('user_id')
        data_train = pd.DataFrame()
        data_test = pd.DataFrame()
        user_reviews = defaultdict(list)  # 字典  uid:List(reviews)
        item_reviews = defaultdict(list)  # 字典  iid:List(reviews)
        user_rid = {}  # 记录user 评论过的item_id
        item_rid = {}
        for user_id, record in user_groupby:
            # print(user_id)
            # print(record.shape[0])
            num = record.shape[0]
            a = record.iloc[:num - 1]
            data_train = data_train.append(a)
            for i in a.values:
                if i[0] in user_reviews.keys():
                    user_reviews[int(i[0])].append(i[3])
                    user_rid[i[0]].append([i[1], float(i[2])])
                else:
                    user_rid[i[0]] = [[i[1], float(i[2])]]
                    user_reviews[int(i[0])] = [i[3]]
                if i[1] in item_reviews.keys():
                    item_reviews[i[1]].append(i[3])
                    item_rid[int(i[1])].append([i[0], float(i[2])])
                else:
                    item_reviews[int(i[1])] = [i[3]]
                    item_rid[int(i[1])] = [[i[0], float(i[2])]]
            b = record.iloc[-1]
            #print(b[0], b[1], b[2])
            data_test = data_test.append(b)
            if b[0] not in user_reviews.keys():
                #print("data_test 没有user id")
                user_rid[j[0]] = [[0, 0]]
                user_reviews[int(b[0])] = [0]
            if b[1] not in item_reviews.keys():
                #print("data_test 没有item id")
                item_reviews[int(b[1])] = [0]
                item_rid[int(b[1])] = [[0, 0]]


        print(data_train.shape)
        print(data_test.shape)


        n_ratings = tp_rating.shape[0] #数据集数量
        # test = np.random.choice(n_ratings, size=int(0.10 * n_ratings), replace=False)
        # test_idx = np.zeros(n_ratings, dtype=bool)
        # test_idx[test] = True
        # tp_test = tp_rating[test_idx]  # 测试集test，uid，iid，rate #数据选一些数据做测试集
        # tp_train = tp_rating[~test_idx] #训练集

        tp_test = data_test[['user_id', 'item_id', 'ratings']]
        tp_train = data_train[['user_id', 'item_id', 'ratings']]

       # data_test = data_test  # 测试集评论加评分数据
        #data_train = data_train  # 训练集评分加评分数据
        # usercount, itemcount = self.get_count(data_train, 'user_id'), self.get_count(data_train, 'item_id')
        print("训练集出现的user，item数量")
        print(usercount, itemcount)
        tp_train.to_csv(os.path.join(self.TPS_DIR, self.args["data"]+'_train.csv'), index=False, header=None)
        tp_test.to_csv(os.path.join(self.TPS_DIR, self.args["data"]+'_test.csv'), index=False, header=None)

        print(item_reviews[11])
        #print(len(user_reviews.keys()))
        #print(len(item_reviews.keys()))
        json.dump(user_reviews, open(os.path.join(self.TPS_DIR, self.args["data"]+'_user_review.json'), 'w+'))
        json.dump(item_reviews, open(os.path.join(self.TPS_DIR, self.args["data"]+'_item_review.json'), 'w+'))
        pickle.dump(user_rid, open(os.path.join(self.TPS_DIR, self.args["data"]+'_user_rid'), 'wb+'))
        pickle.dump(item_rid, open(os.path.join(self.TPS_DIR, self.args["data"]+'_item_rid'), 'wb+'))

        self.usercount, self.itemcount = self.get_count(data,'user_id'), self.get_count(data, 'item_id')  # 计算每一个user,item 数据集，不过只是训练集的数据
        print("last" , itemcount.shape[0])
        print(np.sort(np.array(usercount.values)))
        print(np.sort(np.array(itemcount.values)))



    def process_data(self):
        user_review = os.path.join(self.TPS_DIR, self.args["data"]+'_user_review.json')
        item_review = os.path.join(self.TPS_DIR, self.args["data"]+"_item_review.json")
        train_data = os.path.join(self.TPS_DIR,self.args["data"]+'_train.csv')
        test_data = os.path.join(self.TPS_DIR,self.args["data"]+'_test.csv')
        u_text, i_text, y_train, y_test, vocabulary, vocabulary_inv, uid_train, iid_train, uid_test, iid_test, user_num, item_num = \
            self.load_data(train_data, test_data,user_review, item_review)
        user_rid_file = open(os.path.join(self.TPS_DIR, self.args["data"]+'_user_rid'), "rb")

        item_rid_file = open(os.path.join(self.TPS_DIR, self.args["data"]+'_item_rid'), "rb")
        user_rid = dict(pickle.load(user_rid_file))
        item_rid = dict(pickle.load(item_rid_file))
        user_rid, item_rid = self.build_user_item_rid(user_rid, item_rid)  # 这里就是对user_rid字典
        user_r_id, item_r_id, user_r_rating, item_r_rating = self.spilt_user_item_rid(user_rid, item_rid)
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        userid_train = uid_train[shuffle_indices]  # train文件打乱,user分离开了
        itemid_train = iid_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
        y_train = y_train[:, np.newaxis]  # 扩展多一维
        y_test = y_test[:, np.newaxis]
        userid_train = userid_train[:, np.newaxis]
        itemid_train = itemid_train[:, np.newaxis]
        userid_test = uid_test[:, np.newaxis]
        itemid_test = iid_test[:, np.newaxis]
        batches_train = list(zip(userid_train, itemid_train, y_train))
        batches_test = list(zip(userid_test, itemid_test, y_test))
        train_output = open(os.path.join(self.TPS_DIR, self.args["data"] + '.train'), 'wb+')  # music.train 是(uid,iid,rate)元组
        pickle.dump(batches_train, train_output)
        train_output.close()
        test_output = open(os.path.join(self.TPS_DIR, self.args["data"] +'.test'), 'wb+')
        pickle.dump(batches_test, test_output)
        test_output.close()
        #user_r_id_file = open(os.path.join(self.TPS_DIR, self.args["dara"] + 'use')
        self.args["vocabulary_num"] = len(vocabulary)
        self.args["user_num"] = user_num
        self.args["item_num"] = item_num
        para = {}
        para["user_rid"] = user_rid
        para["item_rid"] = item_rid
        para['user_num'] = user_num
        para['item_num'] = item_num
        #para['user_length'] = u_text[0].shape[0]  #
        # print("user_num:{}".format(u_text[0].shape[0]))
        #para['item_length'] = i_text[0].shape[0]
        #print("item_num:{}".format(i_text[0].shape[0]))
        para["vocabulary_num"] = len(vocabulary)
        para['train_length'] = len(y_train)
        para['test_length'] = len(y_test)
        para['user_review'] = u_text  # 字典 uid:矩阵
        para['item_review'] = i_text
        para["user_r_id"] = user_r_id
        para["item_r_id"] = item_r_id
        para["user_r_rating"] = user_r_rating
        para["item_r_rating"] = item_r_rating
        para_output = open(os.path.join(self.TPS_DIR, self.args["data"]+'.para'), 'wb+')
        pickle.dump(para, para_output)



    def clean_str(self,string):
        """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(r"[^A-Za-z]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def pad_sentences(self,u_text, u_len, padding_word="<PAD/>"): # 可以保留
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        :argument u_text users_review字典
        :argument u_len 最长的评论
        :argument padding_word 填充词
        :return u_text2 填充好的users_review字典
        """
        sequence_length = u_len  # u_len应该是最长的那个
        u_text2 = {}  # 字典形式
        print(len(u_text))
        for i, sentences in enumerate(u_text):
            # print i
            temp = []
            for sentence in sentences:
                if sequence_length > len(sentence):
                    num_padding = sequence_length - len(sentence)
                    new_sentence = " ".join(sentence) + [padding_word] * num_padding
                    temp.append(new_sentence)
                else:
                    new_sentence = sentence[:sequence_length]
                    temp.append(new_sentence)
            u_text2[i] = temp

        return u_text2

    def build_vocab(self,sentences):  # 保留
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  # 字典id对应
        return [vocabulary, vocabulary_inv]

    def build_input_data(self,u_text, i_text, vocabulary):  # 返回的是字典key:np.array(d_max,s_max)
        """
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        # l = len(u_text)
        # print ("u_text type:{}".format(u_text))  每个用户50*100 矩阵
        d_max =  self.args["review_max_num"]
        s_max =  self.args["review_max_word"]
        u_text2 = {}
        for i in u_text.keys():
            u_reviews = u_text[i]
            u = np.zeros(shape=(d_max, s_max))
            for k, review in enumerate(u_reviews):
                if k == d_max - 1:
                    break
                b = []
                for words in review:
                    for word in words.split():
                        if word in vocabulary.keys():
                            b.append(vocabulary[word])
                u[k, :len(b)] = np.array(b[:s_max])  # 每一条评论转为字典id
            u_text2[i] = u
        print("u_text2[1]", u_text2[1])
        i_text2 = {}
        for j in list(i_text.keys()):
            i_reviews = i_text[j]
            i = np.zeros(shape=(d_max, s_max))
            for k, review in enumerate(i_reviews):
                if k == d_max - 1:
                    break
                b = []
                for words in review:
                    for word in words.split():
                        if word in vocabulary.keys():
                            b.append(vocabulary[word])
                i[k, :len(b)] = np.array(b[:s_max])
            i_text2[j] = i
        return u_text2, i_text2

    def build_user_item_rid(self,user_rid, item_rid):
        #用户评论过的商品id和评分
        #目标商品的用户id和评分
        d_max = self.args["review_max_num"]
        user_rid_2 = {}
        item_rid_2 = {}
        for key, items in user_rid.items():
            u_id = key
            u_item2 = np.zeros(shape=(d_max, 2))
            u_items = np.array(items, dtype="int")
            u_item2[:len(u_items)] = u_items[:d_max]
            user_rid_2[u_id] = u_item2
        for key, items in item_rid.items():
            i_id = key
            i_item2 = np.zeros(shape=(d_max, 2))
            i_items = np.array(items, dtype="int")
            i_item2[:len(i_items)] = i_items[:d_max]
            item_rid_2[i_id] = i_item2

        return user_rid_2, item_rid_2

    def spilt_user_item_rid(self,user_rid, item_rid):
        user_r_id = {}  # 用户评论对应的itemid
        item_r_id = {}  # 商品评论对于的userid
        user_r_rating = {}  # 用户评论对应的itemid 的评分情况
        item_r_rating = {}  # 商品评论对应的 评分情况
        for key, item in user_rid.items():
            user_r_id[key] = item[:, 0]
            user_r_rating[key] = item[:, 1]
        for key, item in item_rid.items():
            item_r_id[key] = item[:, 0]
            item_r_rating[key] = item[:, 1]
        return user_r_id, item_r_id, user_r_rating, item_r_rating

    def build_rev_input_data(self,train_rev_file, vocabulary):
        f1 = open(train_rev_file, "rb+")
        train_rev = dict(pickle.load(f1))
        s_max = self.args["review_max_word"]
        train_rev_pro = dict()
        for key, review in train_rev.items():
            a = np.zeros(shape=(s_max))
            b = []
            review = self.clean_str(str(review))
            for word in review.split():
                if word in vocabulary.keys():
                    b.append(vocabulary[word])
            a[:len(b)] = np.array(b[:s_max])
            train_rev_pro[key] = a  # 每一条评论转为字典id
        return train_rev_pro

    def load_data_and_labels(self,train_data, test_data, user_review, item_review):  # 这里就变成长文章了
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        f_train = open(train_data, "r")
        f_test = open(test_data,"r")
        f1 = open(user_review, "r")  # 这里就是训练集的评论不包含验证集的评论
        f2 = open(item_review, "r")  #  item训练集的评论不包含验证集的评论
        user_reviews = dict(json.load(f1))
        item_reviews = dict(json.load(f2))
        # print (type(user_reviews))
        # print (user_review[1])
        print("user_reviews:user_count", len(user_reviews.items()))
        print("item_reviews:item_count", len(item_reviews.items()))
        uid_train = []
        iid_train = []
        y_train = []
        u_text = defaultdict(list)
        i_text = defaultdict(list)
        for line in f_train:
            line = line.split(',')
            uid_train.append(int(line[0]))
            iid_train.append(int(line[1]))
            y_train.append(float(line[2]))

        for key, review in user_reviews.items():
            review = [self.clean_str(str(x)) for x in review]
            u_text[int(key)].append(review)
        print("u_text:user_count", len(u_text.keys()))
        for key, review in item_reviews.items():
            review = [self.clean_str(str(x)) for x in review]
            i_text[int(key)].append(review)
        print("i_text:item_count", len(i_text.keys()))
        # print(type(i_text))
        uid_test = []
        iid_test = []
        y_test = []

        # 这里要考虑验证集可能有训练集没有的userid和itemid
        for line in f_test:
            line = line.split(',')
            uid_test.append(int(float(line[0])))
            iid_test.append(int(float(line[1])))
            y_test.append(float(line[2]))
            if int(float(line[0])) in u_text:  # 验证集的评论是不能用
                continue
            else:
                u_text[int(float(line[0]))].append('<PAD/>')
            if int(float(line[1])) in i_text:
                continue
            else:
                i_text[int(float(line[1]))].append('<PAD/>')

        # print("len")
        # u = np.array([len(i) for x in u_text.values() for i in x])
        # u_len = np.max(u)  #评论过的数量
        # i_len = np.max(i) #取最大的评论数
        # print("u_len_max:", u_len)
        user_num = len(u_text.keys())
        item_num = len(i_text.keys())

        print("user_num:", user_num)
        print("item_num:", item_num)
        return [u_text, i_text, y_train, y_test, uid_train, iid_train, uid_test, iid_test, user_num, item_num]
    def filter_stopword(self,string):
        word_list = word_tokenize(string)
        stop_words = set(stopwords.words('english'))
        word = []
        for i in word_list:
            if i not in stop_words:
                word.append(i)
        return word

    def load_data(self, train_data, test_data, user_review, item_review):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        u_text, i_text, y_train, y_test, uid_train, iid_train, uid_test, iid_test, user_num, item_num = \
            self.load_data_and_labels(train_data, test_data, user_review, item_review)  # 这里u_text还是字典,文本形式

        print("load data done")
        f1 = open(user_review, "r")  # 这里就是训练集的评论不包含验证集的评论
        user_reviews = dict(json.load(f1))
        #stop_words = set(stopwords.words('english'))

        user_voc = [self.filter_stopword(x) for (i, values) in user_reviews.items() for x in values]

        #item_voc = [str(x).split() for (i, values) in user_reviews.items() for x in values]
        vocabulary, vocabulary_inv = self.build_vocab(user_voc)
        print("vocabulary:{}".format(len(vocabulary)))
        u_text, i_text = self.build_input_data(u_text, i_text, vocabulary)  # 长文本转出字典的id

        y_train = np.array(y_train)  # rate分数
        y_test = np.array(y_test)
        uid_train = np.array(uid_train)  # list uid列
        uid_test = np.array(uid_test)
        iid_train = np.array(iid_train)
        iid_test = np.array(iid_test)

        return [u_text, i_text, y_train, y_test, vocabulary, vocabulary_inv,
                uid_train, iid_train, uid_test, iid_test, user_num, item_num]

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
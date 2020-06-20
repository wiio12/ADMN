import tensorflow as tf
import pickle
import time
import os
from tensorflow.python.keras.layers import Dense, Embedding, Conv2D, Dropout, Masking
from tensorflow.python.keras.regularizers import l1, l2
import numpy as np

#原版
class ADMN():
    def __init__(self,args):
        tf.set_random_seed(0)
        np.random.seed(2019)
        #模型基本参数
        self.review_max_word = args["review_max_word"]
        self.review_max_num = args["review_max_num"]  #评论窗口
        self.vocabulary_num = args["vocabulary_num"]
        self.user_num = args["user_num"]
        self.item_num = args["item_num"]
        self.regularizers = args["regularizers"]
        self.rating_weight = args["rating_weight"]  #计算评分的方法
        #用户id向量，文本，商品编码维度 ，一般用户和商品维度要一致
        self.word_embedding_dimension = args["word_embedding_dimension"]
        self.user_embedding_dimension = args["user_embedding_dimension"]
        self.item_embedding_dimension = args["item_embedding_dimension"]
        #cnn卷积层参数
        self.cnn_filters = args["cnn_filters"]
        self.cnn_padding = args["cnn_padding"]
        self.cnn_activation = args["cnn_activation"]
        self.cnn_kernel_regularizer = args["cnn_kernel_regularizer"]
        self.cnn_kernel_size = args["cnn_kernel_size"]
        self.cnn_strides = args["cnn_strides"]
        self.dropout_size  = args["dropout_size"]
        #fm层参数
        self.fm_size = args["fm_size"]
        self.fm_K = args["fm_K"]
        #训练参数
        self.learning_rate = args["learning_rate"]
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.epsilon = args["epsilon"]
        #self.word_embedding_path = os.path.join(args["root_path"],args["input_data_type"],"word_emb.pkl")
        self.batch_size = args["batch_size"]
        self.train_time = args["train_time"]
        self.sess = args["sess"]
        self.is_sample = args["is_sample"]
        self.sample_ratio = args["sample_ratio"]

        with tf.name_scope("creat_placeholder"):
            # shape（none）对应batch大小
            self.user_id = tf.placeholder(dtype="int32", shape=(None, 1), name="user_id")  # user_id

            self.item_id = tf.placeholder(dtype="int32", shape=(None, 1), name="item_id")  # item_id

            self.user_review = tf.placeholder(tf.float32, [None, self.review_max_num , self.review_max_word],
                                          name="user_review")  # user_review 用户评论
            self.item_review = tf.placeholder(tf.float32, [None,  self.review_max_num , self.review_max_word], name="item_review")  # 商品评论

            self.user_commented_items_id = tf.placeholder(dtype="int32", shape=(None, self.review_max_num),
                                           name="user_commented_items_id")  # 用户评论过的商品的id

            self.user_commented_items_rate = tf.placeholder(dtype="float32", shape=(None,self.review_max_num),name="user_commented_items_rate")  # 跟上面user_rid对应评论-评分

            self.item_commented_users_id = tf.placeholder(dtype="int32", shape=(None, self.review_max_num), name="item_commented_users_id")  # 商品评论的人的id

            self.item_commented_users_rate = tf.placeholder(dtype="float32", shape=(None, self.review_max_num),name="item_commented_users_rate")  # 商品的评论的人的给的分数

            self.input_y = tf.placeholder(tf.float32,[None, 1], name="input_y")#评分  # item商品评论

        with tf.name_scope("build_review_embedding"):
            self.user_review_flat = tf.reshape(self.user_review,[-1,self.review_max_num*self.review_max_word])
            print("user_review_flat:{}".format(self.user_review_flat.shape))
            self.item_review_flat = tf.reshape(self.item_review,[-1,self.review_max_num*self.review_max_word])
            print("item_review_flat:{}".format(self.item_review_flat.shape))
            self.user_review_mask = Masking(mask_value=0,input_shape=(self.review_max_num,self.review_max_word))(self.user_review_flat)#mask掉0值，忽略0值
            self.item_review_mask = Masking(mask_value=0,input_shape=(self.review_max_num,self.review_max_word))(self.item_review_flat)#忽略商品评论的0值
            self.review_embedding_layer = Embedding(input_dim=self.vocabulary_num,output_dim=self.word_embedding_dimension,input_length=self.review_max_num*self.review_max_num)
            self.user_review_embedding = self.review_embedding_layer(self.user_review_mask)
            self.user_review_embedding = tf.reshape(self.user_review_embedding,shape=[-1, self.review_max_num, self.review_max_word, self.word_embedding_dimension])
            print("user_review_embedding:{}".format(self.user_review_embedding.shape))

            self.item_review_embedding = self.review_embedding_layer(self.item_review_mask)
            self.item_review_embedding = tf.reshape(self.item_review_embedding,shape=[-1, self.review_max_num, self.review_max_word, self.word_embedding_dimension])
            print("item_review_embedding:{}".format(self.item_review_embedding.shape))

            self.user_review_embedding_sentence = tf.reduce_sum(self.user_review_embedding,axis=2)
            print("user_review_embedding_sentence:{}".format(self.user_review_embedding_sentence.shape))
            self.item_review_embedding_sentence = tf.reduce_sum(self.item_review_embedding,axis=2)
            print("item_review_embedding_sentence:{}".format(self.item_review_embedding_sentence.shape))

     #用户商品id向量编码
        with tf.name_scope("build_user_item_id_embedding"):

            self.user_embedding_layer = Embedding(input_dim=self.user_num,output_dim=self.user_embedding_dimension)
            self.user_id_embedding = self.user_embedding_layer(self.user_id)

            self.item_embedding_layer = Embedding(input_dim=self.item_num,output_dim=self.item_embedding_dimension)
            self.item_id_embedding = self.item_embedding_layer(self.item_id)


            self.user_commented_items_id_mask = Masking(mask_value=0)(self.user_commented_items_id)
            self.item_commented_users_id_mask = Masking(mask_value=0)(self.item_commented_users_id)
            self.user_commented_items_id_mask_embedding = self.item_embedding_layer(self.user_commented_items_id_mask)
            self.item_commented_users_id_mask_embedding = self.user_embedding_layer(self.item_commented_users_id_mask)
            print("user_commented_items_id_mask_embedding:{}".format(self.user_commented_items_id_mask_embedding.shape))
            print("item_commented_users_id_mask_embedding:{}".format(self.item_commented_users_id_mask_embedding.shape))


        with tf.name_scope("build_user_item_extra_embedding"):
            if (self.rating_weight == "base"):  # 1
                self.user_commented_items_rate_sum = tf.reduce_sum(self.user_commented_items_rate, axis=1, keepdims=True)
                self.user_commented_items_rate_base = self.user_commented_items_rate / self.user_commented_items_rate_sum
                self.user_commented_items_rate_base_weight = tf.reshape(self.user_commented_items_rate_base,
                                                                         shape=(-1, self.review_max_num, 1))
                self.user_commented_items_weight = self.user_commented_items_rate_base_weight

                self.item_commented_users_rate_sum = tf.reduce_sum(self.item_commented_users_rate, axis=1, keepdims=True)
                self.item_commented_users_rate_base = self.item_commented_users_rate / self.item_commented_users_rate_sum
                self.item_commented_users_rate_base_weight = tf.reshape(self.item_commented_users_rate_base,
                                                                         shape=(-1, self.review_max_num, 1))
                self.item_commented_users_weight = self.item_commented_users_rate_base_weight
            if(self.rating_weight=="softmax"): #2
                self.user_commented_items_rate_softmax = tf.reshape(tf.nn.softmax(self.user_commented_items_rate,axis=1,name="user_commented_item_rate_softmax"),shape=(-1,self.review_max_num,1))
                self.user_commented_items_weight = self.user_commented_items_rate_softmax

                print("user_commented_items_rate_softmax:{}".format(self.user_commented_items_rate_softmax.shape))
                self.item_commented_users_rate_softmax = tf.reshape(tf.nn.softmax(self.item_commented_users_rate,axis=1,name="item_commented_item_rate_softmax"),shape=(-1,self.review_max_num,1))
                print("item_commented_users_rate_softmax:{}".format(self.item_commented_users_rate_softmax.shape))
                self.item_commented_users_weight = self.item_commented_users_rate_softmax

            if(self.rating_weight == "unbias_softmax"): #3
                self.user_commented_items_rate_mean = tf.reduce_mean(self.user_commented_items_rate,axis=1,keepdims=True)
                self.user_commented_items_rate_unbias = self.user_commented_items_rate - self.user_commented_items_rate_mean
                self.user_commented_items_rate_unbias_softmax = tf.reshape(tf.nn.softmax(self.user_commented_items_rate_unbias,axis=1,name="user_commented_items_rate_unbias_softmax"),shape=(-1,self.review_max_num,1))
                self.user_commented_items_weight = self.user_commented_items_rate_unbias_softmax

                self.item_commented_users_rate_mean = tf.reduce_mean(self.item_commented_users_rate,axis=1,keepdims=True)
                self.item_commented_users_rate_unbias = self.item_commented_users_rate - self.item_commented_users_rate_mean
                self.item_commented_users_rate_unbias_softmax = tf.reshape(tf.nn.softmax(self.item_commented_users_rate_unbias,axis=1,name="item_commented_user_rate_unbias_softmax"),shape=(-1,self.review_max_num,1))
                self.item_commented_users_weight = self.item_commented_users_rate_unbias_softmax

            if (self.rating_weight == "abs_unbias"):  # 4
                self.user_commented_items_rate_mean = tf.reduce_mean(self.user_commented_items_rate, axis=1,
                                                                     keepdims=True)
                self.user_commented_items_rate_abs_unbias = tf.abs(
                    self.user_commented_items_rate - self.user_commented_items_rate_mean)
                self.user_commented_items_rate_abs_unbias_sum = tf.reduce_sum(self.user_commented_items_rate, axis=1,
                                                                              keepdims=True)
                self.user_commented_items_rate_abs_unbias_weight = self.user_commented_items_rate / self.user_commented_items_rate_abs_unbias_sum
                self.user_commented_items_weight = tf.reshape(self.user_commented_items_rate_abs_unbias_weight,
                                                              shape=(-1, self.review_max_num, 1))

                self.item_commented_users_rate_mean = tf.reduce_mean(self.item_commented_users_rate, axis=1,
                                                                     keepdims=True)
                self.item_commented_users_rate_abs_unbias = tf.abs(
                    self.item_commented_users_rate - self.item_commented_users_rate_mean)
                self.item_commented_users_rate_abs_unbias_sum = tf.reduce_sum(self.item_commented_users_rate_abs_unbias,
                                                                              axis=1, keepdims=True)
                self.item_commented_users_rate_abs_unbias_weight = self.item_commented_users_rate / self.item_commented_users_rate_abs_unbias_sum
                self.item_commented_users_weight = tf.reshape(self.item_commented_users_rate_abs_unbias_weight,
                                                              shape=(-1, self.review_max_num, 1))
            if(self.rating_weight == "abs_unbias_softmax"): #5
                self.user_commented_items_rate_mean = tf.reduce_mean(self.user_commented_items_rate, axis=1, keepdims=True)
                self.user_commented_items_rate_abs_unbias = tf.abs(self.user_commented_items_rate - self.user_commented_items_rate_mean)
                self.user_commented_items_rate_abs_unbias_softmax = tf.reshape(
                    tf.nn.softmax(self.user_commented_items_rate_abs_unbias, axis=1,
                                  name="user_commented_items_rate_abs_unbias_softmax"), shape=(-1, self.review_max_num, 1))
                self.user_commented_items_weight = self.user_commented_items_rate_abs_unbias_softmax

                self.item_commented_users_rate_mean = tf.reduce_mean(self.item_commented_users_rate, axis=1, keepdims=True)
                self.item_commented_users_rate_abs_unbias = tf.abs(self.item_commented_users_rate - self.item_commented_users_rate_mean)
                self.item_commented_users_rate_abs_unbias_softmax = tf.reshape(
                    tf.nn.softmax(self.item_commented_users_rate_abs_unbias, axis=1,
                                  name="item_commented_user_rate_abs_unbias_softmax"), shape=(-1, self.review_max_num, 1))
                self.item_commented_users_weight = self.item_commented_users_rate_abs_unbias_softmax

            if(self.rating_weight == "no_rating"): #6
                self.user_review_to_itemId = tf.reshape(tf.multiply(self.user_commented_items_id_mask_embedding,self.item_id_embedding),shape=(-1,self.user_embedding_dimension))
                self.user_review_to_itemId_dense = Dense(1,activation="relu")(self.user_review_to_itemId)
                self.user_review_to_itemId_dense = tf.reshape(self.user_review_to_itemId_dense,shape=(-1,self.review_max_num,1))
                print("user_review_to_itemId_dense:{}".format(self.user_review_to_itemId_dense.shape))
                self.user_review_to_itemId_dense_softmax =tf.nn.softmax(self.user_review_to_itemId_dense, axis=1,name="user_review_to_itemId_dense_softmax")
                print("user_review_to_itemId_dense_softmax:{}".format(self.user_review_to_itemId_dense_softmax.shape))
                self.user_review_to_itemId_dense_softmax = tf.reshape( self.user_review_to_itemId_dense_softmax,shape=[-1,self.review_max_num,1])
                self.user_commented_items_weight = self.user_review_to_itemId_dense_softmax

                self.item_review_to_userId = tf.reshape(tf.multiply(self.item_commented_users_id_mask_embedding,self.user_id_embedding),shape=(-1,self.user_embedding_dimension))
                self.item_review_to_userId_dense = Dense(1,activation="relu")(self.item_review_to_userId)
                self.item_review_to_userId_dense = tf.reshape(self.item_review_to_userId_dense,shape=(-1,self.review_max_num,1))
                self.item_review_to_userId_dense_softmax = tf.nn.softmax(self.item_review_to_userId_dense,axis=1,
                                  name="item_review_to_userId_dense_softmax")
                self.item_review_to_userId_dense_softmax = tf.reshape(self.item_review_to_userId_dense_softmax,shape=[-1,self.review_max_num,1])
                self.item_commented_users_weight = self.item_review_to_userId_dense_softmax

            self.user_review_weight = self.user_commented_items_weight * self.user_review_embedding_sentence
            self.item_review_weight = self.item_commented_users_weight * self.item_review_embedding_sentence
            self.user_review_feature = tf.reduce_sum(tf.multiply(self.user_review_weight,self.item_id_embedding),axis=1,keepdims=True)
            self.item_review_feature = tf.reduce_sum(tf.multiply(self.item_review_weight,self.user_id_embedding),axis=1,keepdims=True)
            print("user_review_feature:{}".format(self.user_review_feature.shape))
            print("item_review_feature:{}".format(self.item_review_feature))


        with tf.name_scope("build_item_attention"):

            self.item_attention = tf.matmul(self.user_id_embedding,tf.transpose(self.item_review_embedding_sentence,[0,2,1]))
            self.item_attention = tf.reshape(tf.nn.softmax(self.item_attention),shape=[-1,self.review_max_num,1])
            print("item_attention:{}".format(self.item_attention.shape))
            self.item_feature = self.item_attention * self.item_review_embedding_sentence
            self.item_feature = tf.reduce_sum(self.item_feature,axis=1,keepdims=True)
        with tf.name_scope("build_user_attention"):

            self.user_attention = tf.matmul(self.item_id_embedding,tf.transpose(self.user_review_embedding_sentence,[0,2,1]))
            self.user_attention = tf.reshape(tf.nn.softmax(self.user_attention),shape=[-1,self.review_max_num,1])
            print("user_attention:{}".format(self.user_attention.shape))
            self.user_feature = self.user_attention * self.user_review_embedding_sentence
            self.user_feature = tf.reduce_sum(self.user_feature,axis=1,keepdims=True)

        with tf.name_scope("build_concat_layer"):
            self.user_feature_concat = tf.concat([self.user_id_embedding,self.user_feature,self.user_review_feature],axis=2,name="user_concat")
            self.item_feature_concat = tf.concat([self.item_id_embedding,self.item_feature,self.item_review_feature],axis=2,name="item_concat")
            print("user_feature_concat:{}".format(self.user_feature_concat.shape))
            print("item_feature_concat:{}".format(self.item_feature_concat.shape))
            self.user_feature_dense = Dense(self.user_embedding_dimension,activation="relu")(self.user_feature_concat)
            self.item_feature_dense = Dense(self.item_embedding_dimension,activation="relu")(self.item_feature_concat)
            print("user_feature_dense:{}".format(self.user_feature_dense.shape))
            print("item_feature_dense:{}".format(self.item_feature_dense.shape))


        with tf.name_scope("build_outer_product"):
            self.user_item_matrix = tf.matmul(tf.transpose(self.user_feature_dense,perm=[0,2,1]),self.item_feature_dense)
            self.user_item_matrix = tf.expand_dims(self.user_item_matrix,-1,name="tran3D")


        with tf.name_scope("build_convolution_layer"):
            self.first_layer = Conv2D(filters=self.cnn_filters,kernel_size=self.cnn_kernel_size,strides=self.cnn_strides,padding=self.cnn_padding,activation=self.cnn_activation,
                                      kernel_regularizer=self.cnn_kernel_regularizer)(self.user_item_matrix)
            self.second_layer = Conv2D(filters=self.cnn_filters,kernel_size=self.cnn_kernel_size,strides=self.cnn_strides,padding=self.cnn_padding,activation=self.cnn_activation,
                                      kernel_regularizer=self.cnn_kernel_regularizer)(self.first_layer)
            self.third_layer = Conv2D(filters=self.cnn_filters,kernel_size=self.cnn_kernel_size,strides=self.cnn_strides,padding=self.cnn_padding,activation=self.cnn_activation,
                                      kernel_regularizer=self.cnn_kernel_regularizer)(self.second_layer)
            self.fourth_layer = Conv2D(filters=self.cnn_filters,kernel_size=self.cnn_kernel_size,strides=self.cnn_strides,padding=self.cnn_padding,activation=self.cnn_activation,
                                      kernel_regularizer=self.cnn_kernel_regularizer)(self.third_layer)
            self.fifth_layer = Conv2D(filters=self.cnn_filters,kernel_size=self.cnn_kernel_size,strides=self.cnn_strides,padding=self.cnn_padding,activation=self.cnn_activation,
                                      kernel_regularizer=self.cnn_kernel_regularizer)(self.fourth_layer)
            self.sixth_layer = Conv2D(filters=self.cnn_filters,kernel_size=self.cnn_kernel_size,strides=self.cnn_strides,padding=self.cnn_padding,activation=self.cnn_activation,
                                      kernel_regularizer=self.cnn_kernel_regularizer)(self.fifth_layer)
            self.dropout_layer = Dropout(self.dropout_size)(self.sixth_layer)


        with tf.name_scope("build_prediction"):
            self.final_vector = tf.reshape(self.dropout_layer,shape=[-1,self.cnn_filters])
            self.fm_w0 = tf.Variable(tf.zeros([1]))
            self.fm_W = tf.Variable(tf.truncated_normal([self.cnn_filters]))

            self.fm_V = tf.Variable(tf.random_normal([self.fm_K,self.cnn_filters],stddev=0.01))
            self.linear_terms = tf.add(self.fm_w0,
                                       tf.reduce_sum(
                                           tf.multiply(self.fm_W,self.final_vector),axis=1,keepdims=True
                                       ))

            self.interactions = tf.add(self.fm_w0,tf.reduce_sum(
                tf.subtract(
                    tf.pow(tf.matmul(self.final_vector,tf.transpose(self.fm_V)),2),
                    tf.matmul(tf.pow(self.final_vector,2),tf.transpose(tf.pow(self.fm_V,2)))),
                axis=1,keepdims=True
                )
            )
            self.output = tf.add(self.linear_terms,self.interactions)

            print("output:{}".format(self.output.shape))
            self.error = tf.subtract(self.output, self.input_y)
        with tf.name_scope("train_loss"):

            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.input_y))))

        with tf.name_scope("test_loss"):  #因为测试集没法一次性输入
            self.test_loss = tf.square(tf.subtract(self.output,self.input_y))


    def model_init(self):
        self.init = tf.global_variables_initializer()
        #sess.run(self.word_embedding_matrix.initializer, feed_dict={self.emb_initializer: self.emb})
        self.sess.run(self.init)

    def load_data(self,train_data,test_data,para_file):
        #train_data为music.train
        para_data = pickle.load(open(para_file,"rb"))
        self.test_data = np.array(pickle.load(open(test_data,"rb")))
        self.train_data = np.array(pickle.load(open(train_data,"rb"))) #这个只是用户商品评论数据
        self.users_review =  para_data['user_review']
        self.items_review = para_data['item_review']
        self.user_r_rating = para_data["user_r_rating"]
        self.item_r_rating = para_data["item_r_rating"]
        self.user_r_id = para_data["user_r_id"]
        self.item_r_id = para_data["item_r_id"]


    def search_train_data(self,uid, iid, user_r_id, item_r_id, user_r_rating, item_r_rating):
        data_num = len(uid)
        user_r_id_batch = np.zeros(shape=(data_num, self.review_max_num))
        item_r_id_batch = np.zeros(shape=(data_num, self.review_max_num))
        user_r_rating_batch = np.zeros(shape=(data_num, self.review_max_num))
        item_r_rating_batch = np.zeros(shape=(data_num, self.review_max_num))
        # user_r_id = list(user_r_id)
        # print (user_r_id[2])
        for i, item in enumerate(uid):
            user_r_id_batch[i, :] = user_r_id[int(item)]
            # print (user_r_id)
            user_r_rating_batch[i, :] = user_r_rating[int(item)]
        for i, item in enumerate(iid):
            item_r_id_batch[i, :] = item_r_id[int(item)]
            item_r_rating_batch[i, :] = item_r_rating[int(item)]
        # print ()
        return user_r_id_batch, item_r_id_batch, user_r_rating_batch, item_r_rating_batch

    def model_train(self):
        #self.model_init()
        print("model_train")
        #self.load_test_data()
        self.test_loss_list = []
        #self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.total_optimizer = tf.train.AdamOptimizer(learning_rate =self.learning_rate,beta1=self.beta1,beta2=self.beta2,epsilon=self.epsilon).minimize(self.loss)
        self.train_data_size = len(self.train_data)
        self.model_init()
        print("data_size_train:{}".format(self.train_data_size))
        self.ll = int(self.train_data_size / self.batch_size) + 1
        print("train_time:{}".format(self.ll))
        for epoch in range(self.train_time):
            print("epoch_i:{}".format(epoch))
            train_rmse = []
            self.shuffle_index = np.random.permutation(np.arange(self.train_data_size))
            self.shuffle_data = self.train_data[self.shuffle_index]
            #print("shuffle_data:",self.shuffle_data.shape)
            for batch_num in range(self.ll):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num+1)*self.batch_size,self.train_data_size-1)
                #print("end_index:",end_index)
                data_train = self.shuffle_data[start_index:end_index]
                batch_user_id,batch_item_id,batch_y =  list(zip(*data_train))
                batch_user_review = []
                batch_item_review = []
                for i in range(len(data_train)):
                    batch_user_review.append(self.users_review[batch_user_id[i][0]])
                    batch_item_review.append(self.items_review[batch_item_id[i][0]])
                batch_user_review = np.array(batch_user_review)
                batch_item_review = np.array(batch_item_review)
                batch_user_r_id,batch_item_r_id,batch_user_r_rate,batch_item_r_rate =self.search_train_data(batch_user_id,
                                                                                                       batch_item_id,
                                                                                                       self.user_r_id,
                                                                                                       self.item_r_id,
                                                                                                       self.user_r_rating,
                                                                                                       self.item_r_rating)
                feed_dict = {
                    self.user_id: batch_user_id,
                    self.item_id: batch_item_id,
                    self.input_y: batch_y,
                    self.user_review: batch_user_review,
                    self.item_review: batch_item_review,
                    self.user_commented_items_id: batch_user_r_id,
                    self.user_commented_items_rate: batch_user_r_rate,
                    self.item_commented_users_id: batch_item_r_id,
                    self.item_commented_users_rate: batch_item_r_rate
                }
                _,t_rmse,error = self.sess.run([self.total_optimizer,self.loss,self.error],feed_dict)
                if self.is_sample==True:
                    self.random_sample(batch_user_id,batch_item_id,batch_y,batch_user_r_id,batch_item_r_id,batch_user_r_rate,batch_item_r_rate,error)
                #current_step = tf.train.global_step(self.sess, self.global_step)
                train_rmse.append(t_rmse)
                print("t_rmse:{}".format(t_rmse))
                if batch_num ==(self.ll-1): #预测
                    print("\nEvaluation:")
                    print(batch_num)
                    self.model_test()


    def show_test_result(self):
        print((" test_loss_list:{}".format(self.test_loss_list)))
        self.besr_test_mse = min(self.test_loss_list)
        print("best test_mse:{}".format(self.besr_test_mse ))
        print('end')

    def model_test(self):


        self.test_data_size = len(self.test_data)
        self.ll_test = int(self.test_data_size / self.batch_size) + 1
        test_cost = []
        for batch_num in range(self.ll_test):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num+1)*self.batch_size,self.test_data_size-1)
            data_test = self.test_data[start_index:end_index]
            user_id_test,item_id_test,y_test = list(zip(*data_test))
            user_valid = []
            item_valid = []
            for i in range(len(data_test)):
                user_valid.append(self.users_review[user_id_test[i][0]])
                item_valid.append(self.items_review[item_id_test[i][0]])
            user_valid = np.array(user_valid)
            item_valid = np.array(item_valid)
            user_r_id_batch, item_r_id_batch, user_r_rate_batch, item_r_rate_batch = self.search_train_data(
                user_id_test, item_id_test, self.user_r_id, self.item_r_id, self.user_r_rating, self.item_r_rating)
            feed_dict = {
                    self.user_id: user_id_test,
                    self.item_id: item_id_test,
                    self.input_y: y_test,
                    self.user_review: user_valid,
                    self.item_review: item_valid,
                    self.user_commented_items_id: user_r_id_batch,
                    self.user_commented_items_rate: user_r_rate_batch,
                    self.item_commented_users_id: item_r_id_batch,
                    self.item_commented_users_rate: item_r_rate_batch
                }
            test_loss = self.sess.run([self.test_loss],feed_dict)
            test_cost.append(test_loss)
        total_mse = 0
        for i in test_cost:
            for j in i:
                for k in j:
                    total_mse += k
        final_mse = total_mse/self.test_data_size
        print("test_final_mse:{}".format(final_mse))
        self.test_loss_list.append(final_mse)


    def random_sample(self,user_id,item_id,y,user_r_id, item_r_id, user_r_rate, item_r_rate,loss):
        num = len(user_id)
        np.random.seed(2019)
        loss =np.array(loss).flatten()
        probability = np.exp(loss)/sum(np.exp(loss))
        #print("probability.shape:{}".format(probability.shape))
        #print("probability length:{}".format(len(probability)))
        #print(probability)
        #print("num:{}".format(num))
        sample_ratio = self.sample_ratio
        #print("sample:{}".format(int(num * sample_ratio)))
        index = np.random.choice(num,size=int(num*sample_ratio),replace=False,p = probability)
        s_user_id = np.array(user_id)[index]
        s_item_id = np.array(item_id)[index]
        s_y = np.array(y)[index]
        s_user_r_id = np.array(user_r_id)[index]
        s_item_r_id = np.array(item_r_id)[index]
        s_user_r_rate = np.array(user_r_rate)[index]
        s_item_r_rate = np.array(item_r_rate)[index]
        s_user_review = []
        s_item_review = []
        for i in range(int(num * sample_ratio)):
            s_user_review.append(self.users_review[s_user_id[i][0]])
            s_item_review.append(self.items_review[s_item_id[i][0]])
        feed_dict = {
            self.user_id: s_user_id,
            self.item_id: s_item_id,
            self.input_y: s_y,
            self.user_review: s_user_review,
            self.item_review: s_item_review,
            self.user_commented_items_id: s_user_r_id,
            self.user_commented_items_rate: s_user_r_rate,
            self.item_commented_users_id: s_item_r_id,
            self.item_commented_users_rate: s_item_r_rate
        }
        _, s_t_rmse = self.sess.run([self.total_optimizer, self.loss], feed_dict)
        print( "s_t_rmse:{}".format(s_t_rmse))


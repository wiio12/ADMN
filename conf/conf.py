import os
from tensorflow.python.keras.regularizers import l1, l2
import tensorflow as tf
#配置文件
class args():
    def __init__(self):
        self.args = {}
        self.args["review_max_word"] = 100
        self.args["review_max_num"]  = 50 # 评论窗口
        self.args["vocabulary_num"] = 263065  #默认
        self.args["user_num"] = 5541  #默认
        self.args["item_num"] = 3568  #默认
        self.args["regularizers"] = 0.1
        self.args["rating_weight"]  = "base" # 计算评分的方法
        # 用户id向量，文本，商品编码维度 ，一般用户和商品维度要一致
        self.args["word_embedding_dimension"] = 64
        self.args["user_embedding_dimension"] = 64
        self.args["item_embedding_dimension"] = 64
        # cnn卷积层参数
        self.args["cnn_filters"] = 32
        self.args["cnn_padding"] = "SAME"
        self.args["cnn_activation"] = "relu"
        self.args["cnn_kernel_regularizer"] = l2(0.01)
        self.args["cnn_kernel_size"] = (2,2)
        self.args["cnn_strides"] = (2,2)
        self.args["dropout_size"] = 0.5
        # fm层参数
        self.args["fm_size"] = 32 #跟cnn卷积核一致
        self.args["fm_K"]  = 10  #fm隐藏层k是维度
        # 训练参数
        self.args["learning_rate"] = 0.002
        self.args["beta1"] = 0.9
        self.args["beta2"] = 0.999
        self.args["epsilon"] = 1e-8
        # self.word_embedding_path = os.path.join(args["root_path"],args["input_data_type"],"word_emb.pkl")
        self.args["batch_size"] = 250
        self.args["train_time"] = 2
        session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        #session_conf.gpu_options.allow_growth = True
        self.args["session_conf"] = session_conf
        self.sess = tf.Session(config=session_conf)
        self.args["sess"] = self.sess
        self.args["data"] = "music"  #默认是music数据，dataloader会变为相应的数据集
        self.args["sample_ratio"] = 0.01
        self.args["is_sample"] = False

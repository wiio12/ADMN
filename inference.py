
from conf import conf
from pro_data import dataloader
from pro_data import dataloader1
from Model import ADMN3
from Model import ADMN2
from Model import ADMN
from Model import ADMN5
import pickle
import time
import os
import json



def save_data(data,parameter):
    now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    save_path = os.path.join(os.getcwd(), "Result")
    save_file = os.path.join(save_path, str(parameter["data"]) ,now  + "_result.txt")
    output = open(save_file, 'w+', encoding="utf-8")
    output.write('best_mse:' + str(min(data))+'\n')
    for i in data:
        output.writelines(str(i)+'\n')
    output.writelines("-----------------------------------"+'\n')
    output.writelines("parameter"+'\n')
    for key, item in parameter.items():
        lines = str(key) + ":" + str(item)
        output.writelines(lines + '\n')
    output.writelines(str(type(model)))
    output.close()

def load_item2id():
    #返回id对应的item名字
    with open(os.path.join(path,parameter["data"]+'_item2id'),'rb') as f:
        item2id = dict(pickle.load(f))
    id2item = {}
    for key,item in item2id.items():
        id2item[item] = key
    return id2item

def save_rec_item():
    with open(os.path.join(path,parameter["data"]+"_user_rec1"), 'rb') as f:
        user_rec_item = dict(pickle.load(f))
    id2item =load_item2id()
    rec_item_dict = {}
    for key,item in user_rec_item.items():
        list = []
        for i in item:
            #print(id2item[i])
            list.append(id2item[i])
        rec_item_dict[key] = list
    pickle.dump(rec_item_dict,open(os.path.join(path,parameter["data"]+"_user_rec_item1"),'wb+'))

def print_user_rec_item(user_id):
    with open(os.path.join(path,parameter["data"]+"_user_rec_item"),'rb') as f:
        user_rec_item = dict(pickle.load(f))
    rec_item = user_rec_item[user_id]
    print(rec_item)

if __name__=="__main__":
    a = conf.args()
    #print(b["batch_size"])
    parameter = a.args
    current_path =os.path.join(os.getcwd(),"Data")
    print(current_path)
    path = os.path.join(current_path,parameter["data"]) #
    file_name = parameter["data"] + ".json"
    datahelper = dataloader1.dataloader(path,file_name,parameter)
    #datahelper.load_file()  #数据处理完就不用处理
    #datahelper.process_data()
    #下面为了免去重复数据分析
    parameter["vocabulary_num"] = datahelper.args["vocabulary_num"] #548680
    parameter["user_num"] = datahelper.args["user_num"]  #5541
    parameter["item_num"] = datahelper.args["item_num"]  #3568
    parameter['is_sample'] = False
    parameter['train_time'] = 2

    #model = ADMN.ADMN(parameter)
    #print(type(model))
    pkl_file =os.path.join(current_path,parameter["data"] ,parameter["data"] + '.para')
    train_path = os.path.join(current_path,parameter["data"],parameter["data"]+'.train')
    test_path =  os.path.join(current_path,parameter["data"],parameter["data"]+'.test')
    #model.load_data(train_path,test_path,pkl_file)
    #list = ['base','softmax','unbias_softmax','abs_unbias','abs_unbias_softmax','no_rating']
    model = ADMN5.ADMN(parameter)
    print(type(model))
    model.load_data(train_path, test_path, pkl_file)
    model.model_train()
    model.show_test_result()
    user_rec = dict()  #保存所有用户的推荐商品
    for i in range(parameter["user_num"]):
        result_rec = list(model.inference(i))
        user_rec[i] = result_rec
        if i%100==0 and i>0:
            print("user_id:{}".format(i))
    pickle.dump(user_rec,open(os.path.join(path,parameter["data"]+"_user_rec1"),"wb+"))
    save_rec_item() #将id转化为item名字，并保存起来
    #print_user_rec_item(1) #输入用户id，返回推荐商品
    #print_user_rec_item(4)
    #pickle.dump(self.item2id, open(os.path.join(self.TPS_DIR, self.args["data"] + '_item2id'), "wb+"))
    #pickle.dump(self.user2id, open(os.path.join(self.TPS_DIR, self.args["data"] + '_user2id'), "wb+"))


    #model.model_train()
    #model.show_test_result()
    #result = model.test_loss_list
    #save_data(result, parameter)

    # model.model_train()
    # model.show_test_result()
    # result = model.test_loss_list
    # save_data(result,parameter)





from conf import conf
from pro_data import dataloader
from Model import ADMN
import pickle
import time
import os

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



if __name__=="__main__":
    a = conf.args()
    #print(b["batch_size"])
    parameter = a.args
    current_path =os.path.join(os.getcwd(),"Data")
    print(current_path)
    path = os.path.join(current_path,parameter["data"]) #
    file_name = parameter["data"] + ".json"
    datahelper = dataloader.dataloader(path,file_name,parameter)
    datahelper.load_file()  #数据处理完就不用处理
    datahelper.process_data()
    #下面为了免去重复数据分析
    parameter["vocabulary_num"] = datahelper.args["vocabulary_num"] #548680
    parameter["user_num"] = datahelper.args["user_num"]  #5541
    parameter["item_num"] = datahelper.args["item_num"]  #3568
    parameter['is_sample'] = False

    #model = ADMN.ADMN(parameter)
    #print(type(model))
    pkl_file =os.path.join(current_path,parameter["data"] ,parameter["data"] + '.para')
    train_path = os.path.join(current_path,parameter["data"],parameter["data"]+'.train')
    test_path =  os.path.join(current_path,parameter["data"],parameter["data"]+'.test')
    #model.load_data(train_path,test_path,pkl_file)
    #list = ['base','softmax','unbias_softmax','abs_unbias','abs_unbias_softmax','no_rating']
    list = ['base', 'softmax', 'unbias_softmax', 'abs_unbias', 'abs_unbias_softmax']
    sample_ratio = [0.1,0.2,0.4,0.8]
    #list = ['no_rating']

    for i in list:
        for j in sample_ratio:
            parameter["rating_weight"] = i
            parameter["sample_ratio"] = j
            model = ADMN.ADMN(parameter)
            print(type(model))
            print("model.rating_weight:{}".format(model.rating_weight))
            model.load_data(train_path, test_path, pkl_file)
            model.model_train()
            model.show_test_result()
            result = model.test_loss_list
            save_data(result, parameter)
    # model.model_train()
    # model.show_test_result()
    # result = model.test_loss_list
    # save_data(result,parameter)





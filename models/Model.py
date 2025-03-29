import torch
#pytoch神经网络的模块化接口
import torch.nn as nn
import os
import glob

#继承了一个nn.module类
class Model(nn.Module):

    def __init__(self, name):
        #类继承
        super(Model, self).__init__()
        self.name = name


    def save(self, path, epoch=0):
        # 为当前path拼接路径
        complete_path = os.path.join(path, self.name)
        #如果不存在则创建目录
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        #保存数据到磁盘
        torch.save(self.state_dict(), 
                os.path.join(complete_path,
                    #zfill(num) 用于返回num长度的字符串 默认前补0
                    "model-{}.pth".format(str(epoch).zfill(5))))


    def save_results(self, path, data):
        #抛出异常
        raise NotImplementedError("Model subclass must implement this method.")
        

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            #glob.glob 返回符合规则的文件路径
            model_files = glob.glob(complete_path+"/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)
        #将预训练的模型参数加载到self即当前模型上
        self.load_state_dict(torch.load(mf))



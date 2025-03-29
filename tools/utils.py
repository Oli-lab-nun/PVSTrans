import os

class Log(object):
    def save_train_info(self, epoch,num,alldata,losses):
        """
        loss may contain several parts
        """
        loss = losses
        datanum = num
        dataall = alldata
        root_dir = os.path.abspath('/')
        log_dir = os.path.join(root_dir, 'log') 
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        log_file = os.path.join(log_dir, 'log_train.txt')
        with open(log_file, 'a+') as f:
            f.write('Train <==> Epoch: [{0}]\t'
                    '{num}/{alldata}\t'
                    'Loss {loss:.6f}\n'
                    .format(epoch,num=datanum,alldata=dataall ,loss = loss))

    
    def save_test_info(self, name,epoch, acc,loss,mean):
        if not os.path.exists("log"):
            os.mkdir("log")

        # if not os.path.exists(log_file):
        #     os.mknod(log_file)
        os.path.join("./log", name+'_validate.txt')
        with open("./log/" + name + "_validate.txt", 'a+') as f:
            f.write('Test <==> Epoch: [{:4d}]  Acc:{:.4f}  mean_acc:{:.4f}  Loss:{:.4f} \n'.format(epoch, acc,mean,loss))

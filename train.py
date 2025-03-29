import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
from tools.ImgDataset import MultiviewImgDataset
from models.MVVIT import LessVII, AutoSelectVIT, CONFIGS
from tools.Trainer import Trainer


parser = argparse.ArgumentParser()

parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVVIT")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=12)
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
"""

更改数据集 要修改数据集的path 数据集的类别数

"""
parser.add_argument("-num_classes", type=int, help="number of class", default=40)

parser.add_argument("-train_path", type=str, default="/bjteam/yh/datasets/modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="/bjteam/yh/datasets/modelnet40_images_new_12x/*/test")
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14"],
                    default="ViT-B_16", help="Which variant to use.")
parser.add_argument("-num_views", type=int, help="number of views", default=20)
parser.add_argument("--pretrained_dir", type=str, default="/bjteam/mxy/opt/tiger/minist/ViT-B_16.npz",
                    help="Where to search for pretrained ViT models.")
# lable smoothing
parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value")
parser.set_defaults(train=False)


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


if __name__ == '__main__':
    args = parser.parse_args()

    log_dir = args.name
    # create log folder
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')

    json.dump(vars(args), config_f)
    
    config_f.close()

    
    n_models_train = args.num_models * args.num_views
   
    log_dir = args.name + '_Train_ModelNet40'
    create_folder(log_dir)

    #transformer configuratin
    config = CONFIGS[args.model_type]
    #Patch-View
    model1 = AutoSelectVIT(args.name+"-Patch-View", config, img_size=224, num_views=args.num_views, num_classes=args.num_classes, smoothing_value=0.0)
    model1.load_from(np.load(args.pretrained_dir))

    # model1.transEncoder.load_from(np.load(args.pretrained_dir))
    #子网二
    model2 = LessVII(args.name+"-View-Shape", config, img_size=224, num_views=args.num_views, num_classes=args.num_classes, smoothing_value=0.0)
    model2.load_from(np.load(args.pretrained_dir))

    #训练集
    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train,
                                        num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    #验证集
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_models=0,
                                      num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    #优化器
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    #训练器
    trainer = Trainer(model1, model2, train_loader, val_loader, optimizer1, optimizer2, 'PVSTrans', log_dir, num_classes=args.num_classes, num_views=args.num_views)
    #训练 epoch
    print('num_train_files: ' + str(len(train_dataset.filepaths)))
    print('num_val_files: ' + str(len(val_dataset.filepaths)))
    trainer.train(100)
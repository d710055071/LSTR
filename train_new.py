import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import  DataLoader

from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory
import models.py_utils.misc as utils
from torch.utils.tensorboard import SummaryWriter



tb_writer = SummaryWriter(log_dir="runs/exp",comment="relu6")

def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detection")
    parser.add_argument("--cfg_file", default='LSTR',help="config file", type=str)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--resume-from",default="/mnt/sda/mycode/LSTR/cache/nnet/LSTR/LSTR_latest.pkl",type=str)
    # parser.add_argument("--resume-from",default="/mnt/sda/mycode/LSTR/cache/nnet/LSTR/LSTR_7500.pkl",type=str)

    
    # parser.add_argument("--resume-from",default=None,type=str)
    args = parser.parse_args()
    return args
def convert_data(data,batch_size):
    imgs,labels,masks,idxs ,path= data

    gt_lanes = []
    for index in range(labels.shape[0]):
        tgt_ids = labels[index][:,0] 
        label   = labels[index][tgt_ids > 0]
        label = np.stack([label] * batch_size, axis=0)
        gt_lanes.append(torch.from_numpy(label.astype(np.float32)).cuda())

    imgs    = imgs.cuda()
    # labels  = labels.cuda()
    masks   = masks.cuda()
    # gt_lanes = gt_lanes.cuda()
    args = {
            "xs": [imgs, masks],
            "ys": [imgs, *gt_lanes],
            # "path":path
    }
    return args
def write_tb_log(metric_logger,iteration,nnet):

    tb_writer.add_scalars(main_tag = "all/global_avg",
                         tag_scalar_dict = {
                            "loss":metric_logger.meters['loss'].global_avg,
                            "loss_ce":metric_logger.meters['loss_ce'].global_avg,
                            "loss_lowers":metric_logger.meters['loss_lowers'].global_avg,
                            "loss_uppers":metric_logger.meters['loss_uppers'].global_avg,
                            "loss_curves":metric_logger.meters['loss_curves'].global_avg
                         },
                         global_step = iteration
                         )

    # tb_writer.add_scalars(main_tag = "all/queue_avg",
    #                      tag_scalar_dict = {
    #                         "loss":metric_logger.meters['loss'].avg,
    #                         "loss_ce":metric_logger.meters['loss_ce'].avg,
    #                         "loss_lowers":metric_logger.meters['loss_lowers'].avg,
    #                         "loss_uppers":metric_logger.meters['loss_uppers'].avg,
    #                         "loss_curves":metric_logger.meters['loss_curves'].avg
    #                      },
    #                      global_step = iteration
    #                      )

    # tb_writer.add_scalars(main_tag = "all/current_step",
    #                      tag_scalar_dict = {
    #                         "loss":metric_logger.meters['loss'].value,
    #                         "loss_ce":metric_logger.meters['loss_ce'].value,
    #                         "loss_lowers":metric_logger.meters['loss_lowers'].value,
    #                         "loss_uppers":metric_logger.meters['loss_uppers'].value,
    #                         "loss_curves":metric_logger.meters['loss_curves'].value
    #                      },
    #                      global_step = iteration
    #                      )


    tb_writer.add_scalar("all_loss/global_avg",metric_logger.meters['loss'].global_avg,iteration)
    # tb_writer.add_scalar("all_loss/queue_avg",metric_logger.meters['loss'].avg,iteration)
    # tb_writer.add_scalar("all_loss/current_step",metric_logger.meters['loss'].value,iteration)


    tb_writer.add_scalar("loss_ce/global_avg",metric_logger.meters['loss_ce'].global_avg,iteration)
    # tb_writer.add_scalar("loss_ce/queue_avg",metric_logger.meters['loss_ce'].avg,iteration)
    # tb_writer.add_scalar("loss_ce/current_step",metric_logger.meters['loss_ce'].value,iteration)


    tb_writer.add_scalar("loss_lowers/global_avg",metric_logger.meters['loss_lowers'].global_avg,iteration)
    # tb_writer.add_scalar("loss_lowers/queue_avg",metric_logger.meters['loss_lowers'].avg,iteration)
    # tb_writer.add_scalar("loss_lowers/current_step",metric_logger.meters['loss_lowers'].value,iteration)


    tb_writer.add_scalar("loss_uppers/global_avg",metric_logger.meters['loss_uppers'].global_avg,iteration)
    # tb_writer.add_scalar("loss_uppers/queue_avg",metric_logger.meters['loss_uppers'].avg,iteration)
    # tb_writer.add_scalar("loss_uppers/current_step",metric_logger.meters['loss_uppers'].value,iteration)


    tb_writer.add_scalar("loss_curves/global_avg",metric_logger.meters['loss_curves'].global_avg,iteration)
    # tb_writer.add_scalar("loss_curves/queue_avg",metric_logger.meters['loss_curves'].avg,iteration)
    # tb_writer.add_scalar("loss_curves/current_step",metric_logger.meters['loss_curves'].value,iteration)

    for name, param in nnet.network.named_parameters():
        tb_writer.add_histogram(name, param, iteration)

def run(type,batch_size,dataloader,model,display_iter,iteration,print_iter):
        if type == "train":
            model.train_mode()
        else:
            model.eval_mode()
        metric_logger = utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue())
        for data in tqdm(dataloader):
            # 最后次不够 batch_size
            if data[0].shape[0] != batch_size:
                continue
            args = convert_data(data,batch_size)
            viz_split = type
            save = True if (display_iter and iteration % display_iter == 0) else False
            # save = False
            (set_loss, loss_dict) = getattr(model,type)(iteration, save, viz_split, **args)
            
            # del set_loss

            (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = loss_dict
            loss_value =  set_loss.item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])


            
            if iteration%100 == 0:
                model.save_params(iteration,False,1000)
            if iteration % print_iter == 0:
                import datetime
                print("")
                now = datetime.datetime.now()
                print("{:},lr = {:.3e},{:}".format(now.strftime('%Y-%m-%d %H:%M:%S'),model.optimizer.param_groups[0]["lr"],str(metric_logger)))
                # write log info 
                write_tb_log(metric_logger,iteration,model)
            iteration+=1
        # if type == "train":
        #     lr = model.optimizer.param_groups[0]["lr"] * 0.99
        #     model.set_lr(lr)
        print(str(metric_logger))
        return metric_logger,iteration,model
def main():
    args = parse_args()
    # 获取配置文件路径
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    # 读取配置文件
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    # 更新配置文件名称
    configs["system"]["snapshot_name"] = args.cfg_file
    # 将文件中的配置更新 system_configs中
    system_configs.update_config(configs["system"])
    # 数据名称
    dataset_name = system_configs.dataset
    # 训练 dataset
    train_dataset = datasets[dataset_name](configs["db"],system_configs.train_split)
    # 评估dataset
    val_daaset = datasets[dataset_name](configs["db"],system_configs.val_split)

    batch_size = system_configs.batch_size
    # 定义加载数据
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=args.threads)
    val_dataloader = DataLoader(val_daaset,batch_size=batch_size,shuffle=False,num_workers=1)
    # 定义模型

    nnet = NetworkFactory(flag=True)

    
    nnet.cuda()
    nnet.train_mode()

    display          = system_configs.display


    # 打印log的间隔
    print_iter = 50
    iteration = 0
    min_loss = None
    iteration ,min_loss = nnet.resume_from(args.resume_from,1000)
    nnet.set_lr(0.0001)
    print("*"*50)
    print("start iteration:",iteration)
    print("*"*50)
    for epoch in range(1000):
        print("epoch:",epoch)
        print("*"*50)
        print("start train...")
        print("*"*50)
        metric_logger , iteration, nnet = run("train",batch_size,train_dataloader,nnet,display,iteration,print_iter)
        print("*"*50)
        print("start val...")
        print("*"*50)
        metric_logger , _, nnet = run("val",batch_size,val_dataloader,nnet,1,1,2048)

        update_best_model = False
        current_val_loss = metric_logger.loss.global_avg
        if min_loss is None or min_loss > current_val_loss:
            min_loss = current_val_loss
            update_best_model = True
            print("*"*50)
            print("min_loss = ",min_loss)
            print("*"*50)
        # 保存checkpoint
        print("*"*50)
        print("save checkpoint...")
        print("*"*50)
        nnet.save_params(iteration,update_best_model,min_loss)

    tb_writer.close()
if __name__ == "__main__":
    main()
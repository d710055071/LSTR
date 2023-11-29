import os
import torch
import importlib
import torch.nn as nn
import shutil
from thop import profile, clever_format
from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, iteration, save, viz_split,
                xs, ys, **kwargs):

        preds, weights = self.model(*xs, **kwargs)

        loss  = self.loss(iteration,
                          save,
                          viz_split,
                          preds,
                          ys,
                          **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, flag=False):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        # print("module_file: {}".format(module_file)) # models.CornerNet
        nnet_module = importlib.import_module(module_file)

        self.model   = DummyModule(nnet_module.model(flag=flag))
        self.loss    = nnet_module.loss()
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes)
        self.flag    = flag
        self.best_model_name = None
        # Count total parameters
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("Total parameters: {}".format(total_params))

        # Count MACs when input is 360 x 640 x 3
        # input_test = torch.randn(1, 3, 360, 640).cuda()
        # input_mask = torch.randn(1, 3, 360, 640).cuda()
        # macs, params, = profile(self.model, inputs=(input_test, input_mask), verbose=False)
        # macs, _ = clever_format([macs, params], "%.3f")
        # print('MACs: {}'.format(macs))


        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        elif system_configs.opt_algo == 'adamW':
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError("unknown optimizer")
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.99)
        
    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self,
              iteration,
              save,
              viz_split,
              xs,
              ys,
              **kwargs):
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [y.cuda(non_blocking=True) for y in ys]

        self.optimizer.zero_grad()
        loss_kp = self.network(iteration,
                               save,
                               viz_split,
                               xs,
                               ys,
                               **kwargs)

        loss      = loss_kp[0]
        loss_dict = loss_kp[1:]
        loss      = loss.mean()

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss, loss_dict

    def val(self,
                 iteration,
                 save,
                 viz_split,
                 xs,
                 ys,
                 **kwargs):

        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            torch.cuda.synchronize()

            loss_kp = self.network(iteration,
                                   save,
                                   viz_split,
                                   xs,
                                   ys)
            loss      = loss_kp[0]
            loss_dict = loss_kp[1:]
            loss      = loss.mean()

            return loss, loss_dict

    def test(self, xs, **kwargs):
        with torch.no_grad():
            # xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)
    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration, is_bbox_only=False):
        cache_file = system_configs.snapshot_file.format(iteration)

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)
    def resume_from(self,checkpoint,step_size = 1500):
        start_iter = 0
        min_loss = None
        if checkpoint:
            ckpt = torch.load(checkpoint)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt["start_iter"]
            if "min_val_loss" in ckpt:
                min_loss = ckpt["min_val_loss"]
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.99)
        return start_iter,min_loss
    # def save_params(self, iteration):
    #     cache_file = system_configs.snapshot_file.format(iteration)
    #     print("saving model to {}".format(cache_file))
    #     with open(cache_file, "wb") as f:
    #         params = self.model.state_dict()
    #         torch.save(params, f)
    def save_params(self,iteration,update_best,min_val_loss):
        # check_point
        cache_file = system_configs.snapshot_file.format(iteration)
        # 保存当前chekcpoint
        checkpoint = {
            'start_iter':iteration,
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'min_val_loss':min_val_loss,
        }
        torch.save(checkpoint,cache_file)

        # last link
        latest_file = system_configs.snapshot_file.format("latest")
        # 判断文件链接是否存在
        if os.path.islink(latest_file):
            # 删除链接
            os.remove(latest_file)
        # 重新创建链接
        os.symlink(os.path.basename(cache_file),latest_file)
        if update_best:

            best_filename = system_configs.snapshot_file.format(str(iteration) + "_best")
            
            if self.best_model_name is not None :
                os.remove(system_configs.snapshot_file.format(self.best_model_name))
            self.best_model_name = str(iteration) + "_best"
            with open(best_filename, "wb") as f:
                params = self.model.state_dict()
                torch.save(params, f)
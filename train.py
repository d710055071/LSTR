#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets
import models.py_utils.misc as utils

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("--cfg_file", default='LSTR',help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--freeze", action="store_true")
    # parser.add_argument("--resume-from",default="/mnt/sda/mycode/LSTR/cache/nnet/LSTR/LSTR_latest.pkl",type=str)
    parser.add_argument("--resume-from",default=None,type=str)
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def prefetch_data(db, queue, sample_data):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train(training_dbs, validation_db, resume_from = None,start_iter=0, freeze=False):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    batch_size       = system_configs.batch_size

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    # training_queue   = Queue(1) # 5
    training_queue   = Queue(system_configs.prefetch_size) # 5
    validation_queue = Queue(5)

    # queues storing pinned data for training
    # pinned_training_queue   = queue.Queue(1) # 5
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size) # 5
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data) # "sample.coco"
    sample_data = importlib.import_module(data_file).sample_data
    # print(type(sample_data)) # function

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(flag=True)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)


    start_iter = nnet.resume_from(resume_from)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))

        # nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    header = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    min_loss = None
    with stdout_to_tqdm() as save_stdout:
        for iteration in metric_logger.log_every(tqdm(range(start_iter + 1, max_iteration + 1),
                                                      file=save_stdout, ncols=67),
                                                 print_freq=500, header=header):

            training = pinned_training_queue.get(block=True)
            viz_split = 'train'
            save = True if (display and iteration % display == 0) else False
            (set_loss, loss_dict) \
                = nnet.train(iteration, save, viz_split, **training)
            (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = loss_dict
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=learning_rate)

            del set_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                viz_split = 'val'
                save = True
                validation = pinned_validation_queue.get(block=True)
                (val_set_loss, val_loss_dict) \
                    = nnet.val(iteration, save, viz_split, **validation)
                (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = val_loss_dict
                print('*'*100)
                print('[VAL LOG]\t[Saving training and evaluating images...]')
                print("loss: {:.4f} class_error: {:.4f}".format(loss_value,loss_dict_reduced['class_error']))
                print('*'*100)
                # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                # metric_logger.update(class_error=loss_dict_reduced['class_error'])
                # metric_logger.update(lr=learning_rate)
                nnet.train_mode()

            if iteration % snapshot == 0:
                update_best_model = False
                if min_loss is None or min_loss > loss_value:
                    min_loss = loss_value
                    update_best_model = True
                nnet.save_params(iteration,update_best_model,min_loss)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

            if iteration % (training_size // batch_size) == 0:
                metric_logger.synchronize_between_processes()
                print("Averaged stats:", metric_logger)


    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()
def main():
    args = parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file  # CornerNet
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    dataset = system_configs.dataset  # MSCOCO | FVV
    print("loading all datasets {}...".format(dataset))

    threads = args.threads  # 4 every 4 epoch shuffle the indices
    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    validation_db = datasets[dataset](configs["db"], val_split)

    # print("system config...")
    # pprint.pprint(system_configs.full)
    #
    # print("db config...")
    # pprint.pprint(training_dbs[0].configs)

    print("len of training db: {}".format(len(training_dbs[0].db_inds)))
    print("len of testing db: {}".format(len(validation_db.db_inds)))

    print("freeze the pretrained network: {}".format(args.freeze))

    train(training_dbs, validation_db, args.resume_from, args.start_iter, args.freeze) # 0

import profile

if __name__ == "__main__":
    # profile.run('main()')

    main()
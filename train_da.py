import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from mp.mp import build_hdd_model
from nets.frcnn_da_all import FasterRCNN
from nets.frcnn_training_da_all import (DAFasterRCNNTrainer, get_lr_scheduler,
                                        set_optimizer_lr, weights_init)
from utils_adan.callbacks import EvalCallback, LossHistory
from utils_adan.utils import (get_classes, seed_everything, show_config,
                              worker_init_fn)
from utils_adan.utils_fit_all import fit_one_epoch
from utils_adan.dataloader import FRCNNDataset, frcnn_dataset_collate

'''
The complete code is being organized. After the paper is accepted, we will provide the complete code.
'''
if __name__ == "__main__":
    Cuda = True
    seed = 11
    fp16 = False
    if not fp16:
        fp32 = True
    else:
        fp32 = False
    classes_path = ''
    model_path = ''
    input_shape = [600, 600]
    backbone = ""
    pretrained = False
    anchors_size = [8, 16, 32]
    Init_Epoch = 0
    Freeze_Epoch = 100
    Freeze_batch_size = 2
    UnFreeze_Epoch = 20
    Unfreeze_batch_size = 8
    Freeze_Train = False
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 5e-4 if optimizer_type == "sgd" else 0
    lr_decay_type = 'cos'
    save_period = 10
    save_dir = ''
    map_out_path = ''
    eval_flag = True
    eval_period = 2
    num_workers = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scene = ''
    train_annotation_path = ''
    train_target_annotation_path = ''
    val_target_annotation_path = ''

    class_names, num_classes = get_classes(classes_path)
    domain_text = ['real scene', 'clipart scene']
    instance_text = ['background'] + [classes for classes in class_names]
    lam = 0.01
    seed_everything(seed)
    token = [5, 10, 10, 25]
    model = FasterRCNN(num_classes, Unfreeze_batch_size, domain_text=domain_text, instance_text=instance_text, fp32=fp32,
                       clip_backbone='model_weight/RN50.pt', token=token, anchor_scales=anchors_size, backbone=backbone,
                       pretrained=pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    with open(train_annotation_path, encoding='utf-8') as f:
        train_source_lines = f.readlines()
    with open(train_target_annotation_path, encoding='utf-8') as f:
        train_target_lines = f.readlines()
    with open(val_target_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_source_lines) + len(train_target_lines)
    num_val = len(val_lines)

    show_config(
        classes_path=classes_path, model_path=model_path, input_shape=input_shape,
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // (Unfreeze_batch_size * 2) * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // (Unfreeze_batch_size * 2) == 0:
            raise ValueError('')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False
        model.freeze_bn()
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        batch_size = batch_size * 2
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_source_dataset = FRCNNDataset(train_source_lines, input_shape, train=True)
        train_target_dataset = FRCNNDataset(train_target_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)

        gen_source = DataLoader(train_source_dataset, shuffle=True, batch_size=batch_size // 2, num_workers=num_workers,
                                pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate,
                                worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_target = DataLoader(train_target_dataset, shuffle=True, batch_size=batch_size // 2, num_workers=num_workers,
                                pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate,
                                worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        train_util = DAFasterRCNNTrainer(model_train, model.clip_model, optimizer, Unfreeze_batch_size, device)

        eval_callback = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda,
                                     map_out_path=map_out_path, eval_flag=eval_flag, period=eval_period)


        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size * 2

                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.extractor.parameters():
                    param.requires_grad = True

                model.freeze_bn()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("")

                gen_source = DataLoader(train_source_dataset, shuffle=True, batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate,
                                        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_target = DataLoader(train_target_dataset, shuffle=True, batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate,
                                        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=frcnn_dataset_collate,
                                     worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, lam, epoch, epoch_step,
                          epoch_step_val, gen_source, gen_target, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler,
                          save_period, save_dir, num_classes)

        loss_history.writer.close()

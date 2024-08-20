import os
from itertools import cycle

import torch
from tqdm import tqdm
from utils.utils import get_lr


def fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, lam, epoch, epoch_step, epoch_step_val,
                  gen_source, gen_target, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, num_classes):
    total_loss = 0
    total_detect_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    _domain_loss = 0
    _global_text_loss = 0
    _instance_text_loss = 0
    _instance_cos_loss = 0
    _svd_loss = 0
    val_loss = 0
    print('Start Train')
    if len(gen_source) <= len(gen_target):
        short_dataset = gen_source
        long_dataset = gen_target
    else:
        short_dataset = gen_target
        long_dataset = gen_source
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, (source_batch, target_batch) in enumerate(zip(long_dataset, cycle(short_dataset))):
            if iteration >= epoch_step:
                break
            images_source, boxes, labels = source_batch[0], source_batch[1], source_batch[2]
            images_target, boxes_target, labels_target = target_batch[0], target_batch[1], target_batch[2]
            with torch.no_grad():
                if cuda:
                    images_source = images_source.cuda()
                    images_target = images_target.cuda()
            detect_loss, domain_classifier_loss, global_text_loss, instance_text_loss, instance_cos_loss, svd_loss = \
                train_util.train_step(lam, num_classes+1, images_source, images_target, boxes, labels, boxes_target, labels_target, 1, fp16, scaler)

            rpn_loc, rpn_cls, roi_loc, roi_cls, total_detect = detect_loss
            total = total_detect + svd_loss
            total_loss += total.item()
            total_detect_loss += total_detect.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss += roi_loc.item()
            roi_cls_loss += roi_cls.item()
            _domain_loss += domain_classifier_loss.item()
            _global_text_loss += global_text_loss.item()
            _instance_text_loss += instance_text_loss.item()
            _instance_cos_loss += instance_cos_loss.item()
            _svd_loss += svd_loss.item()
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'detect_loss': total_detect_loss / (iteration + 1),
                                'rpn_loc': rpn_loc_loss / (iteration + 1),
                                'rpn_cls': rpn_cls_loss / (iteration + 1),
                                'roi_loc': roi_loc_loss / (iteration + 1),
                                'roi_cls': roi_cls_loss / (iteration + 1),
                                '1_domain_loss': _domain_loss / (iteration + 1),
                                '2_global_text': _global_text_loss / (iteration + 1),
                                '3_instance_text': _instance_text_loss / (iteration + 1),
                                '4_instance_cos': _instance_cos_loss / (iteration + 1),
                                '5_svd': _svd_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()

                train_util.optimizer.zero_grad()
                _, _, _, _, val_total = train_util.forward_source(boxes, labels, 1, images, stage='val')
                val_loss += val_total.item()

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

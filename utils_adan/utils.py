import math
import random

import cv2
import numpy as np
import torch
from PIL import Image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width


def gamma_correct(img, mean_source, mean_target):
    # blueImg = img[:, :, 0]
    # greenImg = img[:, :, 1]
    # redImg = img[:, :, 2]
    blueImg_t = img[:, :, 0]
    greenImg_t = img[:, :, 1]
    redImg_t = img[:, :, 2]
    mean_b = (mean_source[0] + mean_target[0]) / 2
    mean_g = (mean_source[1] + mean_target[1]) / 2
    mean_r = (mean_source[2] + mean_target[2]) / 2
    gamma_b = math.log10(0.5) / math.log10(mean_b / 255)
    gamma_g = math.log10(0.5) / math.log10(mean_g / 255)
    gamma_r = math.log10(0.5) / math.log10(mean_r / 255)
    image_gamma_correct_b = gamma_trans(blueImg_t, gamma_b)
    image_gamma_correct_g = gamma_trans(greenImg_t, gamma_g)
    image_gamma_correct_r = gamma_trans(redImg_t, gamma_r)
    new_Img = np.array([image_gamma_correct_b, image_gamma_correct_g, image_gamma_correct_r])
    new_Img = new_Img.transpose((1, 2, 0))
    return new_Img


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def pseudo_label(img_path, results, save_path):
    line = img_path
    with open(save_path, 'a') as f:
        if isinstance(results, int):
            line = line + '\n'
            f.write(line)
            return
        image = Image.open(img_path)
        top_boxes = results[:, :4]
        top_label = np.array(results[:, 5])
        for i in range(len(results)):
            line = line + ' '
            top, left, bottom, right = top_boxes[i]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            line = line + str(left) + ',' + str(top) + ',' + str(right) + ',' + str(bottom) + ','
            line = line + str(int(top_label[i]))
        line = line + '\n'
        f.write(line)




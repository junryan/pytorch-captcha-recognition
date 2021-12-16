# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import torch
import time
from torch.autograd import Variable
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN


def main():
    print('开始对图片进行预测')
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("加载神经网络训练的模型.")
    result = []
    predict_dataloader = my_dataset.get_predict_data_loader()
    for i, (image_name, images, labels) in enumerate(predict_dataloader):
        start = time.time()
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        res = '%s%s%s%s' % (c0, c1, c2, c3)
        cost = '%.2f ms' % ((time.time() - start) * 1000)
        result.append([image_name[0],res, cost])
    print('经过训练后的神经网络预测图片的结果为:')
    data = np.hstack([result])
    res = pd.DataFrame(data, columns=['图片名称', '预测结果', '耗费时间'])
    print(res)


if __name__ == '__main__':
    main()

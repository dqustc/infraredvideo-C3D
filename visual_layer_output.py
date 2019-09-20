from keras import backend as K
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"    # 使用第一, 三块GPU
from keras.models import load_model, model_from_json
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import videoto3d

def main():
    # model = load_model('ucf101-dq/2019-09-08 16:24:35_74_10_0.8191161358466741_ucf101_3dcnnmodel.hd5')
    with open('ucf101-dq/2019-09-12 13:52:44_101_10_ucf101_3dcnnmodel.json', 'r') as model_json:
        model = model_from_json(model_json.read(),custom_objects={'tf': tf})
    model.load_weights('ucf101-dq/2019-09-12 13:52:44_101_10_ucf101_3dcnnmodel.hd5')
    model.summary()
    vid3d = videoto3d.Videoto3D(64, 64, 10, 'files_in_one_dir')
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    # images=cv2.imread("../Project/1.jpg")
    images = []
    images.append(vid3d.video3d('/data/nfs/UCF-101/v_JugglingBalls_g09_c01.avi', color=True, skip=True))
    images = np.array(images).transpose((0, 2, 3, 1, 4))
    # cv2.imshow("Image", images)
    # cv2.waitKey(0)

    # Turn the image into an array.
    # 根据载入的训练好的模型的配置，将图像统一尺寸
    # image_arr = cv2.resize(images, (70, 70))

    # image_arr = np.expand_dims(image_arr, axis=0)

    # 第一个 model.layers[0],不修改,表示输入数据；
    # 第二个model.layers[ ],修改为需要输出的层数的编号[]
    # layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
    layer_mask = K.function([layer_dict['input_1'].input], [layer_dict['multiply_1'].output])
    # 只修改inpu_image

    f1 = layer_mask([images])[0]

    # 第一层卷积后的特征图展示，输出是（1,66,66,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    f1 = f1.squeeze(axis=0)
    for _ in range(5):
        show_img = f1[:, :, _, :]
        # show_img = show_img.squeeze(axis=2)
        plt.subplot(2, 5, _ + 1)
        # plt.imshow(show_img, cmap='black')
        # plt.imshow(show_img, cmap='gray')
        plt.imshow(show_img.astype(np.int))
        plt.subplot(2, 5, 5+_+1)
        plt.imshow(np.array(images[:,:,:,_,:]).squeeze(axis=0))
        plt.axis('off')
    # plt.subplot(2,6,12)
    # plt.imshow(np.array(images[:,:,:,5,:]).squeeze(axis=0))
    plt.show()
    plt.figure(2)
    for _ in range(5):
        show_img = f1[:, :, 5+ _, :]
        # show_img = show_img.squeeze(axis=2)
        plt.subplot(2, 5, _ + 1)
        # plt.imshow(show_img, cmap='black')
        # plt.imshow(show_img, cmap='gray')
        plt.imshow(show_img.astype(np.int))
        plt.subplot(2, 5, 5 + _ + 1)
        plt.imshow(np.array(images[:, :, :, 5+ _, :]).squeeze(axis=0))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
    input('press any key\n')
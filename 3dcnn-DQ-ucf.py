import argparse
import os
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"    # 使用第一, 三块GPU
import time
from time import sleep
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
# from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, add, Reshape,RepeatVector,BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU, ReLU, Softmax
from keras.losses import categorical_crossentropy
from keras.layers.merge import concatenate, Multiply
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
import videoto3d
from tqdm import tqdm
from keras.layers.core import Lambda
import tensorflow as tf

def is_empty(any_structure):
    if any_structure:
        # print('Structure is not empty.')
        return False
    else:
        # print('Structure is empty.')
        return True

def mean_filter(x):
    x = K.mean(x,axis=[1,2,3])
    return x

def reshape_filter(x, shape):
    x = K.reshape(x, shape=shape)
    return x

def repeat_filter(x, rep, axis):
    x = K.repeat_elements(x,rep,axis=axis)
    return x

def bi_trans(x, th):
    x = K.greater(x, th)
    return tf.cast(x, dtype='float32')

def normalize(x):
    a = K.max(x, axis=[1,2], keepdims=True)
    return x/a

def my_loss(y_true, y_pred):

    pixels1= K.sum(K.cast(K.greater(y_pred, 0.5), 'float32'))
    pixels2= K.sum(K.cast(K.less_equal(y_pred, 0.5), 'float32'))
    mask1 = Multiply()([K.cast(K.greater(y_pred, 0.5), 'float32'), y_pred])  # values greater than 0.5
    mask0 = Multiply()([K.cast(K.less_equal(y_pred, 0.5), 'float32'), y_pred])  # values greater than 0.5

    return -K.log((K.sum(mask1)/pixels1-K.sum(mask0)/pixels2)/255)

def unlikely_loss(y_true, y_pred):
    # norm = K.sqrt(K.sum(K.square(y_pred))) / y_pred.shape[0] * y_pred..
    e_x = K.exp(y_pred-K.expand_dims(K.max(y_pred,axis = 1)))
    p_x = e_x / K.expand_dims(K.sum(e_x, axis=1))
    return -K.mean(K.sum(y_true * K.log(1-p_x+0.0000001), axis=1))

current_time = time.strftime('%Y-%m-%d %H:%M:%S')
def plot_history(history, result_dir):
    plt.plot(history.history['dense_2_acc'], marker='.')
    plt.plot(history.history['val_dense_2_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['dense_2_loss'], marker='.')
    plt.plot(history.history['val_dense_2_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['dense_2_loss']
    acc = history.history['dense_2_acc']
    loss3 = history.history['dense_3_loss']
    acc3 = history.history['dense_3_acc']
    val_loss = history.history['val_dense_2_loss']
    val_acc = history.history['val_dense_2_acc']
    val_acc3 = history.history['val_dense_3_acc']
    val_loss3 = history.history['val_dense_3_loss']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\tloss_n\tacc_n\tval_loss_n\tval_acc_n\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i],loss3[i],acc3[i],val_loss3[i],val_acc3[i]))


def loaddata(video_dir, vid3d, nclass, result_dir, dataset, depth, color=False, skip=True):
    files = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []

    if dataset == 'files_in_one_dir':
        pbar = tqdm(total=len(files))
    else:
        files_num = 0
        for subdir in files:
            files_num += len(os.listdir(os.path.join(video_dir,subdir)))
        pbar = tqdm(total=files_num)

    if dataset == 'files_in_one_dir' :
        for filename in files:
            pbar.update(1)
            if filename == '.DS_Store':
                continue
            name = os.path.join(video_dir, filename)
            statinfo = os.stat(name)
            if statinfo.st_size == 0:
                continue
            label = vid3d.get_UCF_classname(filename)
            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)
            labels.append(label)
            X.append(vid3d.video3d(name, color=color, skip=skip))
    else:
        for dirname in files:
            label = vid3d.get_HMDB_classname(dirname)
            subdir = os.path.join(video_dir, dirname)
            subfiles = os.listdir(subdir)
            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)
            for filename in subfiles:
                pbar.update(1)
                name = os.path.join(subdir, filename)
                statinfo = os.stat(name)
                if statinfo.st_size == 0:
                    continue
                images_frame = vid3d.video3d(name, color=color, skip=skip)

                if len(images_frame) != 0:
                    nclip = np.floor(images_frame.shape[0] / depth).astype(np.int)
                    for n in range(nclip):
                        labels.append(label)
                        X.append(images_frame[n*depth:(n+1)*depth, :,:,:])

    pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 1, 4)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels


def myGenerator(X_train, X_test, Y_train, Y_test, nb_classes, nb_batch):
    # img_rows = 224
    # img_cols = 224

    # Y_train = np_utils.to_categorical(Y_train, nb_classes)

    # X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 10, 3)

    # X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 10, 3)

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')

    # X_train /= 255
    # X_test /= 255

    t = X_train.shape[0] / nb_batch

    steps = np.floor(t).astype(np.int)

    while 1:
        for i in range(steps):  # 1875 * 32 = 60000 -> # of training samples
            if i % 125 == 0:
                print(i)
            yield X_train[i * nb_batch:(i + 1) * nb_batch], [Y_train[i * nb_batch:(i + 1) * nb_batch],Y_train[i * nb_batch:(i + 1) * nb_batch]]


def main():
    import pynvml
    pynvml.nvmlInit()
    # 这里的0是GPU id
    handle2 = pynvml.nvmlDeviceGetHandleByIndex(2)
    handle3 = pynvml.nvmlDeviceGetHandleByIndex(3)
    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    # print(meminfo.used)

    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--dataset',type=str, default='ucf101')
    args = parser.parse_args()

    img_rows, img_cols, frames = 64, 64, args.depth
    channel = 3 if args.color else 1
    fname_npz = 'dataset_{}_{}_{}_{}.npz'.format(
        args.dataset, args.nclass, args.depth, args.skip)

    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames, args.dataset)
    nb_classes = args.nclass

    if os.path.exists(fname_npz):
        loadeddata = np.load(fname_npz)
        X, Y = loadeddata["X"], loadeddata["Y"]
    else:
        x, y = loaddata(args.videos, vid3d, args.nclass,
                        args.output, args.dataset, frames,args.color, args.skip)
        X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        Y = np_utils.to_categorical(y, nb_classes)

        X = X.astype('float32')
        np.savez(fname_npz, X=X, Y=Y)
        print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))


    # Define model

    # conv3D + Relu + Conv3D + Softmax + Pooling3D + DropOut
    input_x = Input(shape = (img_rows, img_cols, frames, channel))

    #
    # # C3D-conv1
    # convLayer = Conv3D(32, kernel_size= (3, 3, 3),padding='same')(input_x)
    # convLayer = ReLU()(convLayer)
    #
    # convLayer = Conv3D(32, kernel_size= (3, 3, 3), padding='same')(convLayer)
    # convLayer = Softmax()(convLayer)
    # convLayer = MaxPooling3D(pool_size=(3,3,3), padding='same')(convLayer)
    # convLayer = Dropout(0.25)(convLayer)
    #
    # # C3D-conv2
    # convLayer = Conv3D(64, kernel_size= (3, 3, 3),padding='same')(convLayer)
    # convLayer = ReLU()(convLayer)
    #
    # convLayer = Conv3D(64, kernel_size= (3, 3, 3), padding='same')(convLayer)
    # convLayer = Softmax()(convLayer)
    # convLayer = MaxPooling3D(pool_size=(3,3,3), padding='same')(convLayer)
    # convLayer = Dropout(0.25)(convLayer)
    #
    #
    # maskLayer = Conv3D(64*frames, kernel_size=(3,3,2), padding='same')(convLayer)
    # maskLayer = Lambda(mean_filter)(maskLayer)      # [None,1, 64], each point represent a mask of input region of 8x8 points
    # # maskLayer = BatchNormalization()(maskLayer)
    # maskLayer = Lambda(K.sigmoid)(maskLayer)
    # # maskLayer = ReLU()(maskLayer)
    # # maskLayer = Lambda(bi_trans, arguments={'th':0.5})(maskLayer)
    # maskLayer = Reshape(( 8, 8, frames, 1))(maskLayer)  #reshape_filter(maskLayer, shape=[None,8,8,1,1])
    # # maskLayer = Lambda(normalize)(maskLayer)
    # maskLayerForLoss = maskLayer
    # maskLayer = Lambda(repeat_filter,arguments={'rep':8, 'axis':1})(maskLayer)
    # maskLayer = Lambda(repeat_filter,arguments={'rep':8, 'axis':2})(maskLayer)
    # # maskLayer = Lambda(repeat_filter,arguments={'rep':frames, 'axis':3})(maskLayer)
    # maskLayer = Lambda(repeat_filter,arguments={'rep':channel, 'axis':4})(maskLayer)
    #
    # # maskLayer = Lambda(repeat_filter,arguments={'rep':2, 'axis':3})(maskLayer)
    # # maskLayer = Lambda(repeat_filter,arguments={'rep':64, 'axis':4})(maskLayer)
    #
    #
    # convLayer = Multiply()([maskLayer,input_x])
    #
    #


    # C3D-conv1
    convLayer = Conv3D(32, kernel_size= (3, 3, 3),padding='same')(input_x)
    convLayer = ReLU()(convLayer)

    convLayer = Conv3D(32, kernel_size= (3, 3, 3), padding='same')(convLayer)
    convLayer = Softmax()(convLayer)
    convLayer = MaxPooling3D(pool_size=(3,3,3), padding='same')(convLayer)
    convLayer = Dropout(0.25)(convLayer)

    # C3D-conv2
    convLayer = Conv3D(64, kernel_size= (3, 3, 3),padding='same')(convLayer)
    convLayer = ReLU()(convLayer)

    convLayer = Conv3D(64, kernel_size= (3, 3, 3), padding='same')(convLayer)
    convLayer = Softmax()(convLayer)
    convLayer = MaxPooling3D(pool_size=(3,3,3), padding='same')(convLayer)
    convLayer = Dropout(0.25)(convLayer)


    fc1 = Flatten()(convLayer)

    fc = Dense(512, activation='sigmoid')(fc1)
    fc = Dropout(0.5)(fc)
    dense_out = Dense(nb_classes, activation='softmax')(fc)
    dense_out_converse = Dense(nb_classes)(fc)

    # model = Model(input_x, [dense_out, dense_out_converse])
    model = Model(input_x, [dense_out, dense_out_converse])

    # loss of 2 parts
    losses = {
        'dense_2':K.categorical_crossentropy,
          'dense_3':unlikely_loss
    }
    lossWeights = {'dense_2':1,
                   'dense_3':1
                   }
    model.compile(loss=losses,
                  loss_weights=lossWeights,
                  optimizer=Adam(lr=0.001), metrics=['accuracy'])
    # model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001),metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True,
               to_file=os.path.join(args.output, 'model.png'))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=43)
    X_train, X_val,Y_train,Y_val = train_test_split(X_train,Y_train, test_size=0.1, random_state=43)

    # history = model.fit_generator(myGenerator(X_train, X_test, Y_train, Y_test, nb_classes, args.batch),
    #                               samples_per_epoch=X_train.shape[0], epochs=args.epoch, verbose=1,
    #                               callbacks=callbacks_list,
                                  # shuffle=True)
    # check GPUs status , once a GPU is available, change os environment parameters and break
    # is none of the GPUs are ready ,sleep for 2 secs and retry.
    cnt = 0
    while True:
        cnt+=1
        processinfo = pynvml.nvmlDeviceGetComputeRunningProcesses(handle2)
        if len(processinfo) == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '2'
            print('GPU 2 is available, use GPU 2\n')
            break
        processinfo = pynvml.nvmlDeviceGetComputeRunningProcesses(handle3)
        if len(processinfo) == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '3'
            print('GPU 3 is available, use GPU 3\n')
            break
        sleep(2)
        print('\rretry time: {}'.format(cnt), end='')


    history = model.fit(X_train, [Y_train, Y_train], validation_data=(X_val, [Y_val,Y_val]), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)
    # history = model.fit(X_train, Y_train,
    #                     validation_data=(X_val, Y_val),
    #                     batch_size=args.batch,
    #                     epochs=args.epoch, verbose=1, shuffle=True)
    # loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, '{}_{}_{}_ucf101_3dcnnmodel.json'.format(current_time,nb_classes,args.depth)), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, '{}_{}_{}_ucf101_3dcnnmodel.hd5'.format(current_time,nb_classes,args.depth)))
    loss = model.evaluate(X_test, [Y_test, Y_test], verbose=0)
    # loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    plot_history(history, args.output)
    save_history(history, args.output)

    print('Test loss:', loss)
    # print('Test accuracy:', acc)


if __name__ == '__main__':
    main()
    input("trainding done, press any key!\n")

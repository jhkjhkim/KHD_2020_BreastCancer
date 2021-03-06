import os
import argparse
import sys
import time
import numpy as np
from matplotlib.image import imread
import tensorflow as tf  # Tensorflow 2
# import arch
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
import math
from arch import cnn, cnn_base, build_xception, build_xception2,build_densenet, build_inception, build_resnet50, recall, precision, f1, sp, ntv, custom, cust_loss_function
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import Xception
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from tensorflow.keras.layers import experimental


######################## DONOTCHANGE ###########################
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')


    def infer(image_path):
        result = []
        X = PathDataset(image_path, labels=None, batch_size=batch_size)
        y_hat = model.predict(X)


        result.extend(np.argmax(y_hat, axis=1))

        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)


def path_loader(root_path):
    image_path = []
    image_keys = []
    for _, _, files in os.walk(os.path.join(root_path, 'train_data')):
        for f in files:
            path = os.path.join(root_path, 'train_data', f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader(root_path, keys):
    labels_dict = {}
    labels = []
    with open(os.path.join(root_path, 'train_label'), 'rt') as f:
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels


############################################################


class PathDataset(tf.keras.utils.Sequence):
    def __init__(self, image_path, labels=None, batch_size=32, test_mode=True):
        self.image_path = image_path
        self.labels = labels
        self.mode = test_mode
        self.batch_size = batch_size
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    def __getitem__(self, idx):
        image_paths = self.image_path[idx * self.batch_size:(idx + 1) * self.batch_size]

        # resize & rescale
        batch_x = np.array([resize(imread(x), (299, 299)) for x in image_paths])

        if not self.mode:

            batch_list = []

            for i in range(len(image_paths)):
                theta, shear = np.random.uniform(-20, 20), np.random.uniform(-20, 20)
                flip_v, flip_h = np.random.randint(0, 2), np.random.randint(0, 2)
                zoom_z, zoom_y = np.random.uniform(0.8, 1), np.random.uniform(0.8, 1)
                x_transform = self.datagen.apply_transform(batch_x[i],
                                                      transform_parameters={'theta': theta,
                                                                            'zx': zoom_z,
                                                                            'zy': zoom_y,
                                                                            'flip_vertical': flip_v,
                                                                            'flip_horizontal': flip_h,
                                                                            'shear': shear})

                batch_list.append(x_transform)
                # if i == 0:
                #    print("i am here2")
            batch_x = np.array(batch_list)

        #if np.max(batch_x[0]) > 1:
        #    batch_x = batch_x / 255.
        #    print("image has a pixel larger than 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        ### REQUIRED: PREPROCESSING ###

        if self.mode:
            return batch_x
        else:
            batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
            #y_test = tf.keras.utils.to_categorical(y_test, num_classes)
            return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.image_path) / self.batch_size)

class PathDataset2(tf.keras.utils.Sequence):
    def __init__(self, image_path, labels=None, batch_size=32, test_mode=True):
        self.image_path = image_path
        self.labels = labels\


        self.mode = test_mode
        self.batch_size = batch_size
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    def __getitem__(self, idx):
        image_paths = self.image_path[idx * self.batch_size:(idx + 1) * self.batch_size]

        # resize & rescale
        batch_x = np.array([resize(imread(x), (299, 299)) for x in image_paths])

        ### REQUIRED: PREPROCESSING ###

        if self.mode:
            return batch_x
        else:
            batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
            return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.image_path) / self.batch_size)


if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    args = argparse.ArgumentParser()

    ########### DONOTCHANGE: They are reserved for nsml ###################
    args.add_argument('--mode', type=str, default='train', help='submit?????? ???????????? test??? ???????????????.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork ???????????? ??????????????? ?????????????????? ???????????????. ??????????????? ????????? ????????? ????????? wall time ??? model ??? ???????????????.')
    args.add_argument('--pause', type=int, default=0, help='model ??? load ?????? 1??? ???????????????.')
    ######################################################################

    # hyperparameters
    args.add_argument('--epoch', type=int, default=500)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--learning_rate', type=int, default=0.001)

    config = args.parse_args()

    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = 2
    learning_rate = config.learning_rate

    # model setting ## ????????? ??? ???????????? ???????????????

    model = build_xception()


    # Loss and optimizer

    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', recall, precision, f1, sp, ntv, custom])


    ############ DONOTCHANGE ###############
    bind_model(model)
    if config.pause:  ## test mode ????????? ????????? ??????
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train':  ### training mode ????????? ????????? ??????
        print('Training Start...')
        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH, 'train')
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################
        '''
        nsml.load(checkpoint='18', session='KHD032/Breast_Pathology/383')
        nsml.save('save')

        model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy', recall, precision, f1, sp, ntv, custom])
        #exit()
        '''
        skf = StratifiedKFold(n_splits=10, random_state=10)
        labels = np.asarray(labels)
        # model number

        target = 3

        count = 0
        for train_index, test_index in skf.split(image_path, labels):
            image_path_trn, image_path_val = image_path[train_index], image_path[test_index]
            labels_trn, labels_val = labels[train_index], labels[test_index]
            if count == target:
                break
            count += 1


        unique, counts = np.unique(labels_trn, return_counts=True)
        num_trn = dict(zip(unique, counts))
        print("Number of Train Class", num_trn)

        unique, counts = np.unique(labels_val, return_counts=True)
        num_val = dict(zip(unique, counts))
        print("Number of Val Class", num_val)

        X = PathDataset(image_path_trn, labels_trn, batch_size=batch_size, test_mode=False)
        #X_train = PathDataset(image_path_trn, labels_trn, batch_size=batch_size, test_mode=True)
        X_val = PathDataset2(image_path_val, labels_val, batch_size=batch_size, test_mode=False)



        ####  to find threshold

        #for i in range(25):
        #    x_test, y_test = X_val.__getitem__(i)
        #    y_hat = model.predict(x_test)
        #    y_prob = y_hat.copy()

        #    y_hat = np.argmax(y_hat, axis=1)


            #print("y_hat.shape", y_hat.shape)
            #print("type(y_hat)", type(y_hat))
         #   print("abs(y_test - y_hat)", abs(y_test - y_hat))
          #  if abs(y_test - y_hat).any():
           #     print("miss classified!!!!!!!!!")
           #     print("true label     :", y_test)
           #     print("predicted label:",y_hat)
           #     print("predicted probability:", y_prob)

           # else:
            #    continue




        #print("X_train.__getitem__(0)", X_train.__getitem__(0))
        #print("model.predict(X_train.__getitem__(0)", model.predict(X_train.__getitem__(0)))

        #print("X.__getitem__(0)", X.__getitem__(0))

        print("---------------it is working----------------")



        #input_ = tf.keras.Input(shape=(32, 32, 3))
        #m1 = model(input_)
        #out = tf.keras.layers.concatenate([m1, 1 - m1])
        #ensemble = tf.keras.models.Model(inputs=input_, outputs=out)
        #ensemble.summary()

        #best = 10
        #cnt = 0
        # patience
        #patience = 2

        #for epoch in range(num_epochs):
        #    hist = model.fit(X, validation_data=X_val, shuffle=True)
        #    val_loss = hist.history['val_loss'][-1]
        #    if best > val_loss:
        #        print(":::best val loss updated")
        #        best = val_loss
        #        cnt = 0
        #    else:
        #        cnt += 1
        #        print(':::not updated, count', cnt)
        #        if cnt >= patience:
        #            model.optimizer.lr = model.optimizer.lr / 5
        #            print(':::**learning rate decreased to', model.optimizer.lr.numpy())
        #    nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=hist.history['loss'])  # , acc=train_acc)
        #    nsml.save(epoch)

        best = 10
        cnt = 0
        #patience
        patience = 3

        for epoch in range(num_epochs):
            print("current epoch:", epoch)


            hist = model.fit(X, validation_data=X_val ,shuffle=True)
            val_loss = hist.history['loss'][-1]
            if best > val_loss:
                print(":::best val loss updated")
                best = val_loss
                cnt = 0
            else:
                cnt += 1
                print(':::not updated, count', cnt)
                if cnt >= patience:
                    model.optimizer.lr = model.optimizer.lr / 5
                    print(':::**learning rate decreased to', model.optimizer.lr.numpy())

            """
            if epoch < 20:
                model.optimizer.lr = model.optimizer.lr * 0.95
                print("::::current learning rate:", model.optimizer.lr.numpy())
            else:
                model.optimizer.lr = model.optimizer.lr * 0.9
                print("::::current learning rate:", model.optimizer.lr.numpy())
            
            """

            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=hist.history['loss'])  # , acc=train_acc)
            nsml.save(epoch)
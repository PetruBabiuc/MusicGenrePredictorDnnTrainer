# -*- coding: utf-8 -*-
import tensorflow
import tflearn
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import config

def create_model(nb_classes, img_size) -> DNN:
    initializer_creator = tensorflow.keras.initializers.GlorotUniform
    activation = 'relu'

    print("[+] Creating model...")
    convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

    convnet = conv_2d(convnet, 64, 2, activation=activation, weights_init=initializer_creator())
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation=activation, weights_init=initializer_creator())
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation=activation, weights_init=initializer_creator())
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation=activation, weights_init=initializer_creator())
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation=activation)
    # 0.5 initial
    convnet = dropout(convnet, 0.2)

    convnet = fully_connected(convnet, nb_classes, activation='softmax')
    convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

    model = tflearn.DNN(convnet,
                        checkpoint_path='DNN saves/Last training, all checkpoints/musicDNN.tflearn',
                        best_checkpoint_path='DNN saves/Last train, best checkpoint/musicDNN.tflearn',
                        # best_val_accuracy=0.3
                        )
    print("    Model created! ✅")
    return model


def save_model(model: DNN):
    # Save trained model
    print("[+] Saving the weights...")
    model.save('DNN saves/Last training, last epoch/musicDNN.tflearn')
    print("[+] Weights saved! ✅💾")


def load_model(model: DNN):
    # Load weights
    print("[+] Loading weights...")
    model.load(config.dnn_path)
    print("    Weights loaded! ✅")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from glob import glob
from random import shuffle
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.applications.densenet import DenseNet201
from keras.optimizers import Adam, SGD, RMSprop
from imgaug import augmenters as iaa
import imgaug as ia


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10),  # rotate by -45 to +45 degrees
                shear=(-5, 5),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(3, 5)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 5)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                           ]),
                           iaa.Invert(0.01, per_channel=True),  # invert color channels
                           iaa.Add((-2, 2), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-1, 1)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.9, 1.1), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-1, 0),
                                   first=iaa.Multiply((0.9, 1.1), per_channel=True),
                                   second=iaa.ContrastNormalization((0.9, 1.1))
                               )
                           ]),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    return seq


# aug test file for predict
def data_gen_test(list_files, batch_size, augment=False):
    seq = get_seq()
    while True:
        # shuffle(list_files)  # shuffle must be closed when testing.
        for batch in chunker(list_files, batch_size):
            X = [cv2.imread(x) for x in batch]
            if augment:
                X = seq.augment_images(X)
            X = [preprocess_input(x) for x in X]
            yield np.array(X)


def dn121_model(input_shape=(96, 96, 3), include_top=False):
    inputs = Input(input_shape)
    base_model = DenseNet201(weights=None, include_top=include_top, input_shape=input_shape)
    x = base_model(inputs)
    out = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation="sigmoid", name="outputs")(out)
    model = Model(inputs=inputs, outputs=outputs)
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model



def main():

    # load test data
    test_files = glob('./data/test/*.tif')
    test_file_num = len(test_files)

    # load weights
    model = dn121_model()
    weight_path = "./data/weights/model.h5"
    model.load_weights(weight_path)

    # aug 5times for predict
    aug_times = 5
    for i in range(aug_times):
        test_generator = data_gen_test(test_files, batch_size, augment=True)
        if i == 0:
            pred_res = model.predict_generator(test_generator, test_file_num)
        else:
            pred_res *= model.predict_generator(test_generator, test_file_num)

    pred_res = pred_res**(1/aug_times)

    # image id
    test_id = [i.lstrip("../input/histopathologic-cancer-detection/test/").rstrip(".tif") for i in test_files]

    # save submit
    df = pd.DataFrame({'id': test_id, 'label': pred_res})
    df.to_csv("dn121_aug5.csv", index=False)



if __name__ == "__main__":
    batch_size = 32
    main()
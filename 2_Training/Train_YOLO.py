import os
import sys
import argparse
import warnings

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

src_path = os.path.join(get_parent_dir(0), "src")
sys.path.append(src_path)

utils_path = os.path.join(get_parent_dir(1), "Utils")
sys.path.append(utils_path)

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from PIL import Image 
from time import time
import tensorflow.compat.v1 as tf
import pickle
from Train_Utils import get_classes,get_anchors,create_model,data_generator_wrapper


keras_path = os.path.join(src_path, "keras_yolo3")
Data_Folder = os.path.join(get_parent_dir(1), "dataset")
Image_Folder = os.path.join(Data_Folder, "images")
data_filename = os.path.join("data_train.txt")

Model_Folder = os.path.join(get_parent_dir(1), 'Data', "Model_Weights")
YOLO_classname = os.path.join(Model_Folder, "data_classes.txt")

anchors_path = os.path.join(keras_path, "model_data", "yolo_anchors.txt")
weights_path = os.path.join(keras_path, "yolo.h5")

if __name__ == "__main__":
    
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, height, width
    epoch1, epoch2 = 51,51

    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path)

    checkpoint = ModelCheckpoint(os.path.join(log_dir, "checkpoint.h5"),monitor="val_loss",save_weights_only=True,save_best_only=True,period=5)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)

    val_split = FLAGS.val_split
    with open(data_filename) as f:
        lines = f.readlines()

    num_val = 30  
    num_train = 250

    # Training with frozen layers first, to get a stable loss.
    frozen_callbacks = [checkpoint]

    model.compile(optimizer=Adam(lr=1e-3),loss={"yolo_loss": lambda y_true, y_pred: y_pred})

    batch_size = 32
    
    print("Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
    
    history = model.fit_generator(
        data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val // batch_size),
        epochs=epoch1,
        initial_epoch=0,
        callbacks=frozen_callbacks,
    )
    model.save_weights(os.path.join(log_dir, "trained_weights_stage_1.h5"))

    # Unfreezing and continue training, to fine-tune.

    full_callbacks = [logging, checkpoint, reduce_lr, early_stopping]

    if _has_wandb:
        full_callbacks.append(wandb_callback)

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(
        optimizer=Adam(lr=1e-4), loss={"yolo_loss": lambda y_true, y_pred: y_pred}
    )  

    print("Unfreezing all layers.")

    batch_size = 4  
    
    print("Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
    
    history = model.fit_generator(
        data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val // batch_size),
        epochs=epoch1 + epoch2,
        initial_epoch=epoch1,
        callbacks=full_callbacks,)

    model.save_weights(os.path.join(log_dir, "trained_weights_final.h5"))

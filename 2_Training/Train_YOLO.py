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
from Train_Utils import get_classes,get_anchors,create_model,create_tiny_model,data_generator_wrapper


keras_path = os.path.join(src_path, "keras_yolo3")
Data_Folder = os.path.join(get_parent_dir(1), "dataset")
Image_Folder = os.path.join(Data_Folder, "images")
VoTT_Folder = os.path.join(Image_Folder)
YOLO_filename = os.path.join(VoTT_Folder, "data_train.txt")

Model_Folder = os.path.join(get_parent_dir(1), 'Data', "Model_Weights")
YOLO_classname = os.path.join(Model_Folder, "data_classes.txt")

log_dir = Model_Folder
anchors_path = os.path.join(keras_path, "model_data", "yolo_anchors.txt")
weights_path = os.path.join(keras_path, "yolo.h5")

FLAGS = None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--annotation_file",type=str,default=YOLO_filename)
    parser.add_argument("--classes_file",type=str,default=YOLO_classname)
    parser.add_argument("--log_dir",type=str,default=log_dir)
    parser.add_argument("--anchors_path",type=str,default=anchors_path)
    parser.add_argument("--weights_path",type=str,default=weights_path)
    parser.add_argument("--val_split",type=float,default=0.1)
    parser.add_argument("--is_tiny",default=False,action="store_true")
    parser.add_argument("--random_seed",type=float,default=None)
    parser.add_argument("--epochs",type=int,default=51)
    parser.add_argument("--warnings",default=False,action="store_true")

    FLAGS = parser.parse_args()

    if not FLAGS.warnings:
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore")

    # Get WandB integration if setup
    try:
        import wandb
        from wandb.integration.keras import WandbCallback  # type: ignore

        wandb.ensure_configured()
        if wandb.api.api_key is None:
            _has_wandb = False
            wandb.termwarn(
                "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable."
            )
        else:
            _has_wandb = False if os.getenv("WANDB_DISABLED") else True
    except (ImportError, AttributeError):
        _has_wandb = False

    np.random.seed(FLAGS.random_seed)

    log_dir = FLAGS.log_dir

    class_names = get_classes(FLAGS.classes_file)
    num_classes = len(class_names)

    if FLAGS.is_tiny and FLAGS.weights_path == weights_path:
        weights_path = os.path.join(os.path.dirname(FLAGS.weights_path), "yolo-tiny.h5")
    else:
        weights_path = FLAGS.weights_path

    if FLAGS.is_tiny and FLAGS.anchors_path == anchors_path:
        anchors_path = os.path.join(
            os.path.dirname(FLAGS.anchors_path), "yolo-tiny_anchors.txt"
        )
    else:
        anchors_path = FLAGS.anchors_path

    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, height, width
    epoch1, epoch2 = FLAGS.epochs, FLAGS.epochs

    is_tiny_version = len(anchors) == 6  # default setting
    if FLAGS.is_tiny:
        model = create_tiny_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path)

    log_dir_time = os.path.join(log_dir, "{}".format(int(time())))
    logging = TensorBoard(log_dir=log_dir_time)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, "checkpoint.h5"),monitor="val_loss",save_weights_only=True,save_best_only=True,period=5)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)

    val_split = FLAGS.val_split
    with open(FLAGS.annotation_file) as f:
        lines = f.readlines()

    num_val = 30  
    num_train = 250

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    frozen_callbacks = [logging, checkpoint]

    if _has_wandb:
        wandb.init(project="TrainYourOwnYOLO", config=vars(FLAGS), sync_tensorboard=False)
        wandb_callback = WandbCallback(save_model=False)
        frozen_callbacks.append(wandb_callback)

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

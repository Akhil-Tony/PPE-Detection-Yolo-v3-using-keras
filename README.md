# Training YOLO v3 using Transfer Learning 
### Model Architecture
Model is created to predict 4 Class Probabilities ['Helmet','Vest','Mask','Boot'].<br>
The Model Uses 9 Anchor Boxes.<br>
The Model used the darknet body as the classification backbone and pretrained imagenet weights were loaded.<br>
A Custom detection head is constructed with 3 layers outputing features with dimentions 13 * 13 * (num_anchors * (num_classes + 5)),<br>
26 * 26 * (num_anchors * (num_classes + 5)),<br>
52 * 52 * (num_anchors * (num_classes + 5)) respectively is stacked upon the classification backbone.
[model.py](https://github.com/Akhil-Tony/PPE-Detection-Yolo-v3-using-keras/blob/c536ab42215577578a84c1b3c29c52ed2c6b96c3/2_Training/src/keras_yolo3/yolo3/model.py#L63-L91)
### Training
The Training is done in two stages
2700 samples from the dataset are used for training and 300 for validation. <br>
First the backbone layers are freezed and the detection heads are trained with learning rate 1e-3 for 10 epochs,
In the second stage for fine tuning the model all the layers are unfreezed and trained with learning rate reduced to 1e-4 for another 10 epochs along with early stopping callback.
[TrainYolo.py](/2_Training/Train_YOLO.py)
Training logs are here [logs](https://github.com/Akhil-Tony/PPE-Detection-Yolo-v3-using-keras/blob/master/Training.ipynb)
### Testing
A video containing workers in construction environment is used for testing.<br>
The Model is loaded and made predictions using [Detection.py](/3_Inference/Detector.py).
<br>
![](https://github.com/Akhil-Tony/PPE-Detection-Yolo-v3-using-keras/blob/master/gif/20220906_133255.gif)
![](https://github.com/Akhil-Tony/PPE-Detection-Yolo-v3-using-keras/blob/master/gif/20220906_133431.gif)
### Dataset Used
[images](https://drive.google.com/drive/folders/1a6HCLloZ0oY1X8Q7rWQkGkITDzZcCDME?usp=sharing)
<br>
[labels](https://drive.google.com/drive/folders/1ews9qncvjQ6aSMuc0rS68SswHLy5X4LV?usp=sharing)

The Downloaded Dataset labels was in class_name x_center,y_center,width,height and N lines for a image.
Using [convert.py](/convert.py) labels were converted to and stored in [data_train.txt](/data_train.txt) with image_file_path [box1]....[boxN]
each box represent x_min,y_min,x_max,y_max,class_index

### Model Weights
[Trained Weights](https://drive.google.com/file/d/1UypC7fhBKwbb9OtTyFFnhZEIkKjbx4mv/view?usp=sharing)
<br>
Download and place inside the Data/Model_Weights Folder for running inference.

### Reference
https://github.com/qqwweee/keras-yolo3

### My Failed Attempt
[first approach notebook](/Yolo_Experiment.ipynb) <br>
I First tried to implement yolo v1 from scratch with my understanding. Since i wanted to use pretrained weights i reused lower layers of VGG19 Model so that i can loaded pretrained image net weights.
then i build a detection head layer for my model by the reshaping the outputs from dense layers.The architecture outputs 7 * 7 * (B*5 + n_classes).

then i build a data pipeline which preprocess each data labels into 7*7*14 label tensor.

Trained with more than 2500 data samples for 100 epochs but model never converged !!!!!


# Training YOLO v3 using Transfer Learning 
### Model Architecture
Model is created to predict 4 Class Probabilities ['Helmet','Vest','Mask','Boot'].<br>
The Model Uses 9 Anchor Boxes.<br>
The Model used the darknet body as the classification backbone and pretrained imagenet weights were loaded.<br>
A Custom detection head is constructed with 3 layers outputing features with dimentions 13 * 13 * (num_anchors * (num_classes + 5)),<br>
26 * 26 * (num_anchors * (num_classes + 5)),<br>
52 * 52 * (num_anchors * (num_classes + 5)) respectively is stacked upon the classification backbone.<br>
Model Building Program : [model.py](https://github.com/Akhil-Tony/PPE-Detection-Yolo-v3-using-keras/blob/c536ab42215577578a84c1b3c29c52ed2c6b96c3/2_Training/src/keras_yolo3/yolo3/model.py#L78)
### Training
2700 samples from the dataset are used for training and 300 for validation. <br>
The Training is done by two stages. <br>
First the backbone layers are freezed and only the detection heads are trained with learning rate 1e-3 for 10 epochs with batch_size 32. <br>
In the second stage for fine tuning the model all the layers are unfreezed and trained with learning rate reduced to 1e-4 for another 10 epochs along with early stopping callback.<br>
Here batch_size is reduced to 4 to not overload the gpu.<br>
Training Program : [TrainYolo.py](/2_Training/Train_YOLO.py)<br>
Total Training Time : 40 mins<br>
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

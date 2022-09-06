# Training YOLO v3 using Transfer Learning 
1. The Downloaded Dataset labels was in class_name x_center,y_center,width,height and N lines for a image.
Using ![convert.py](/convert.py) labels were converted to and stored in ![data_train.txt](/data_train.txt) with image_file_path [box1]....[boxN]
each box represent x_min,y_min,x_max,y_max,class_index
### Model Architecture
The Model uses the darknet body as the classification backbone and imagenet weights are loaded.
A Custom detection head is constructed with 3 layers outputing feature dimentions 13 * 13 * (num_anchors * (num_classes + 5)) , 26 * 26 * (num_anchors * (num_classes + 5)) , 52 * 52 * (num_anchors * (num_classes + 5)) respectively.![model.py](https://github.com/Akhil-Tony/PPE-Detection-Yolo-v3-using-keras/blob/c536ab42215577578a84c1b3c29c52ed2c6b96c3/2_Training/src/keras_yolo3/yolo3/model.py#L63-L91)
### Training
The Training is done in two stages 
First the backbone layers are freezed and the detection heads are trained with learning rate 1e-3 for 51 epochs,
In the second stage all the layers are unfreezed and trained with learning rate reduced to 1e-4 for another 51 epochs along with early stopping callback.

# Training YOLO v3 using Transfer Learning 
1. The Downloaded Dataset labels was in class_name x_center,y_center,width,height and N lines for a image.
Using ![convert.py](/convert.py) labels were converted to and stored in ![data_train.txt](/data_train.txt) with image_file_path [box1]....[boxN]
each box represent x_min,y_min,x_max,y_max,class_index
2. ![model](/2_Training/src/keras_yolo3/yolo3/model.py:43)

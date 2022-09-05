from tqdm import tqdm
import pickle
import cv2
import numpy as np

def read_labels(label_file,shape):
    height,width,_ = shape
    box_and_cls = ''
    for box in label_file.readlines():
        cls,x,y,w,h = box.split(' ')
        cls = int(cls)
        x,y = float(x),float(y)
        w,h = float(w),float(h)
        
        x_min = int( (x - (w/2)) * width )
        x_max = int( (x + (w/2)) * width )
        y_min = int( (y - (h/2)) * height )
        y_max = int( (y + (h/2)) * height )
        
        label = '{},{},{},{},{}'.format(x_min,y_min,x_max,y_max,cls)
        box_and_cls = box_and_cls+' '+label
    return box_and_cls


def process_data_label(files,target_file):
    
    for file in tqdm(files,total=len(files)):
        label_file_path = label_path+file+'.txt'
        image_file_path = image_path+file+'.jpg'
        try:
            image = cv2.imread(image_file_path)
            label_file = open(label_file_path)
        except FileNotFoundError:
            print('no file')
            continue
        
        label_part = read_labels(label_file,image.shape)
        file_line = '/content/drive/MyDrive/'+image_file_path + label_part
        target_file.write(file_line+'\n')
    print('done')
    target_file.close()

if __name__ == '__main__':
    with open('train_files.pkl','rb') as f:
        train_file = pickle.load(f)
    with open('test_files.pkl','rb') as f:
        test_file = pickle.load(f)

    image_path = 'dataset/images/'
    label_path = 'dataset/labels/'

    data_txt = open('data_train.txt', 'w', encoding='utf-8')

    process_data_label(train_file[:3000],data_txt)

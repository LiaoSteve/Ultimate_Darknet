## How to train Darknet YOLOv4
* Yolo v4 paper:    [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)

* Yolo v4 source code:  [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)  
#### 0. Install cuda and cudnn
* [Install CUDA 10.0 and cuDNN v7.4.2 on Ubuntu 16.04](https://gist.github.com/matheustguimaraes/43e0b65aa534db4df2918f835b9b361d)
* [多版本CUDA与cuDNN管理](https://dinghow.site/2019/05/19/cuda-cudnn-version/)


#### 1. Compile on Linux
```
  git clone https://github.com/LiaoSteve/darknet.git
  cd darknet
  gedit Makefile
```
* GPU=1 to build with CUDA to accelerate by using GPU (CUDA should be in /usr/local/cuda)
* CUDNN=1 to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in /usr/local/cudnn)
*  OPENCV=1 to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams.
*  LIBSO=1 to build a library darknet.so and binary runable file uselib that uses this library.
*  Choose your GPU capability (ARCH)
*  Notice that your cuda path NVCC
*  Save and close the Makefile, and type `make` in terminal.
#### 2. Create my own dataset, and label
* [Download in Windows](https://tzutalin.github.io/labelImg/) choose Windows_v1.8.0 and unzip it.
* Create folder :
*  `git clone https://github.com/LiaoSteve/pascal-VOC.git`, and open it.   
*  Delete the `deleteme.txt` inside  these dirs below.
    ```  
    VOCdevkit
    |----VOC2007
        |----Annotations
        |----ImageSets    
        |    |---- Main            
        |----JPEGImages    
    
    ```
* Put your images to JPEGImages dir, and open labelimg dir, open data folder, edit predefined_classes.txt (type your class per line)
* Click labelImg.exe, choose `Open Dir` to `JPEGImages` images, and `Change Save Dir` to `Annotations` dir.
* Use `pascal VOC` label format, and start labeling your images.
* Now if your label work done, put `VOCdevkit` dir and `voc_label.py` into `darknet/data` dir.
* In `data` dir, open terminal and run create_imageSets.py, and check your ImageSets/Main dir, there are four .txt files:
    ```  
    cd VOCdevkit/VOC2007
    python create_imageSets.py    
    ```  
* Open voc_label.py, and revise the code line 7 (`classes = ["class1","class2"]`) to your classes :
    ```    
    cd ../../
    python voc_label.py
    ```
* Now in data dir, you will see `2007_train.txt`, `2007_val.txt`, and there are many .txt YOLO format labels in `VOCdevkit/VOC2007/labels` 
* (Optional) Copy .txt YOLO format labels to images dir :
  ```
  cp -r ./VOCdevkit/VOC2007/labels/*.txt ./VOCdevkit/VOC2007/JPEGImages/
  ```
* Create `obj.names` file in `darknet/data` dir, and type your classes name (each line one class) :
    ``` 
    gedit obj.names
    ```
* Create `obj.data` file :
    ``` 
    gedit obj.data
    ```
* Paste text (revise `classes`) below to `obj.data`:
  
    ```
    classes= 2
    train  = data/2007_train.txt
    valid  = data/2007_val.txt
    names = data/obj.names
    backup = backup/
    ```
#### 3. Download pre-trained weights [yolov4.conv.137](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view), [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74) and put it in `darknet` dir.

#### 4. Edit cfg file. 
* [AlexeyAB darknet README.md](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)
- .cfg : filters=(classes + 5)x3

#### 5. Start training by using the command line: 
* In `darknet` dir :
```
./darknet detector train data/obj.data cfg/yolov4.cfg yolov4.conv.137 -map | tee -a train.log
```
* Or remote version with no GUI :
```
./darknet detector train data/obj.data cfg/my_yolov4.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map
```
#### 6. After training 
* Test image : 
```
./darknet detector test data/obj.data cfg/yolov4.cfg backup/yolov4.weight
```
* Test video and save :
```
./darknet detector demo data/obj.data cfg/yolov4.cfg backup/yolov4.weight video.mp4 -out_filename result.avi
```
* Test webcam :
```
./darknet detector demo data/obj.data cfg/yolov4.cfg backup/yolov4.weight -c 0
```
* Test mAP : 
```
./darknet detector map data/obj.data cfg/yolov4.cfg backup/yolov4.weight
```
* Test recall :
```
./darknet detector recall data/obj.data cfg/yolov4.cfg backup/yolov3.weight
```
#### 8. How to improve object detection : 
* https://github.com/AlexeyAB/darknet#How-to-improve-object-detection
  
### How to create custom dataset from OpenImage Dataset
* [https://github.com/theAIGuysCode/OIDv4_ToolKit](https://github.com/theAIGuysCode/OIDv4_ToolKit)
* Download images and labels :
  ```
  pip install -r requirement.txt
  python main.py downloader --classes Apple --type_csv train --limit 100  
  ```
* Revise `classes.txt`, and run python `convert_annotations.py`
  ```
  python convert_annotations.py
  ```
* Copy images to `VOC2007/JPEG` dir, and copy labels (yolo format .txt) to `labels` dir
* Copy `generate_train.py` to `darknet/data/` dir, run and train yolov4 :
  
  ```  
  python generate_train.py
  
  ./darknet detector train data/obj.data cfg/yolov4.cfg yolov4.conv.137 -map | tee -a train.log
  ```
### How to learn deep learning
* [https://www.youtube.com/watch?v=G_fsA-OUqNw&list=PL1w8k37X_6L9YSIvLqO29S9H0aZ1ncglu&index=1](https://www.youtube.com/watch?v=G_fsA-OUqNw&list=PL1w8k37X_6L9YSIvLqO29S9H0aZ1ncglu&index=1)

## Dairy
* YOLOv3
  
| Num |Batch|subdivision|  Size   |  mAP  |recall |
|:---:|:---:|:---------:|:-------:|:-----:|:-----:|
| 1   | 64  | 16        | 416*416 |       |       | 

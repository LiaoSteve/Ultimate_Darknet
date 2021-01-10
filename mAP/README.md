# mAP for Darknet YOLO Sereis
* Fork from https://github.com/Cartucho/mAP 
* Adapted by Yu-Hsien Liao (LiaoSteve) 

## Getting Started

Create input
```
cd ..
python mAP_create_ground-truth.py
python mAP_create_detections.py
python mAP_copy_images_to_dir.py
cd mAP
```

Create output with images
```
python main.py
```

Create output with no images
```
python main.py -na
```

## Result
```
Special case to draw in:
        - green -> TP: True Positives (object detected and matches ground-truth)

        - red -> FP: False Positives (object detected but does not match ground-truth)

        - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        
        - blue -> ground thruth bbox
```

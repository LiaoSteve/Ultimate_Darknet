# Adapted by LiaoSteve
from ctypes import *
import random
import os
import cv2
import darknet
import argparse

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection") 
    parser.add_argument("--dataset_dir", type=str, default="/home/ti/steve/trash_dataset/data8_crawler/JPEGImages/",help="path to your image set ")

    parser.add_argument("--save_dir", type=str, default="./predict_image/no_crawler500/best/",
                        help="path to save detection images")

    parser.add_argument("--weights", default="./backup/yolov4_8_best.weights",
                        help="yolo weights path") 

    parser.add_argument("--config_file", default="./cfg/yolov4_8.cfg",
                        help="path to config file")

    parser.add_argument("--data_file", default="./data/obj.data",
                        help="path to data file")

    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")    

    parser.add_argument("--iou_thresh", type=float, default=.5,
                        help="nms: remove detections with iou higher this value")    
    
    return parser.parse_args()


def check_arguments_errors(args):    
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    assert 0 < args.iou_thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if not os.path.exists(args.dataset_dir):
        raise(ValueError("Invalid dataset file path {}".format(os.path.abspath(args.dataset_dir))))
    os.makedirs(args.save_dir, exist_ok=1)


if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    darknet_image = darknet.make_image(darknet_width, darknet_height, 3)    

    cur_path = os.getcwd()        
    images = list()

    for filename in os.listdir(args.dataset_dir):
        if filename.endswith('jpg') or filename.endswith('png') \
            or filename.endswith('jpeg') \
            or filename.endswith('JPG') \
            or filename.endswith('PNG') \
            or filename.endswith('JPEG'):
            images.append(filename)
        else:        
            raise RuntimeError(f'notice that {filename} image format are not accepted(.jpg, .png, .jpeg)')
    for image in images:        
        frame = cv2.imread(args.dataset_dir + image)          
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh, nms=args.iou_thresh)
        frame = darknet.draw_boxes(detections, frame, class_colors, darknet_width)
        cv2.imwrite(args.save_dir + 'out_' + image, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #print(f'- [x] save image {image} to {args.save_dir}')
    print(f'- [OK] Save {len(images)} images done')

#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-03-07 08:59:34
@LastEditTime: 2019-03-07 11:13:20
'''

import os
import time
import json

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
from PIL import Image

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

if not os.path.exists('result'):
    os.mkdir('result')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def predict_save(model, test_img_fold, test_img_list):
    # load image
    sub_dict = {}
    sub_dict['result'] = []
    for i in range(2):
    #for i in range(len(test_img_list)):
        # for i in range(1):
        img_name = test_img_list[i]
        img_path = os.path.join(test_img_fold, img_name)
        image = read_image_bgr(img_path)
        
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        image_np = load_image_into_numpy_array(image)    
        re,predictions = coco_demo.run_on_opencv_image(image_np)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        # process image
        start = time.time()
        # print(image.shape)
        # print(scale)
        boxes = predictions.bbox
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        
        print("processing time: ", time.time() - start)
        # correct for image scale
        
        image_dict = {}
        image_dict['filename'] = img_name
        image_dict['rects'] = []
        
        for box, score, label in zip(boxes, scores, labels):
            # scores are sorted so we can break
            if score < 0.5:
                break
            box_dict = {}
            color = 'red' #label_color(label)
            b = [int(i) for i in box]
            score = float(score)
            label = int(label)
            
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
            box_dict['xmin'] = int(b[0])
            box_dict['ymin'] = int(b[1])
            box_dict['xmax'] = int(b[2])
            box_dict['ymax'] = int(b[3])
            box_dict['label'] = int(labels)
            box_dict["confidence"] = int(score)
            
            image_dict['rects'].append(box_dict)
        imsave('result/'+img_name, draw)
        sub_dict['result'].append(image_dict)
    with open("record.json","w") as f:
        json.dump(sub_dict,f)
        print("加载入文件完成...")


if __name__ == "__main__":
    config_file = "../configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x_val.yaml"
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    # print(model.summary())

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'tieke', 1: 'heiding',
                       2: 'daoju', 3: 'dian', 4: 'jiandao'}

    test_img_fold = '/home/hywel/Documents/keras-retinanet/keras_retinanet/CSV/jinnan2_round1_train_20190305/restricted'
    test_img_list = os.listdir(test_img_fold)
    print(len(test_img_list))
    predict_save(model, test_img_fold, test_img_list)
import json
import numpy as np
import pandas as pd
restrict_rele_path = '/home/hywel/Documents/keras-retinanet/keras_retinanet/CSV/jinnan2_round1_train_20190305/restricted/'
label_path = '/home/hywel/Documents/keras-retinanet/keras_retinanet/CSV/jinnan2_round1_train_20190305/train_no_poly.json'
def restrict_image_info(label_path):
    with open(label_path, 'r') as load_f:
        load_dict = json.load(load_f)
        image_collect = load_dict['images']
        image_num = len(image_collect)
        anno_collect = load_dict['annotations']
        anno_num = len(anno_collect)

        img_path_list = []
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        category_list = []

        img_path_val_list = []
        x1_val_list = []
        y1_val_list = []
        x2_val_list = []
        y2_val_list = []
        category_val_list = []

        mapper = {0: 'tieke', 1: 'heiding',
                  2: 'daoju', 3: 'dian', 4: 'jiandao'}

        train_rate = 0.9
        hight = image_num*train_rate
        train_img_id = np.random.randint(0, image_num, size=int(hight))
        print(len(train_img_id))

        for i in range(image_num):
            img = image_collect[i]
            img_name = img['file_name']
            img_id = img['id']
            img_height = img['height']
            img_width = img['width']
            if i in train_img_id:
                for j in range(anno_num):
                    if anno_collect[j]['image_id'] == img_id:
                        bbox = anno_collect[j]['bbox']
                        img_path_list.append(restrict_rele_path+img_name)
                        x1_list.append(int(np.rint(bbox[0])))
                        y1_list.append(int(np.rint(bbox[1])))
                        x2_list.append(
                            int(np.rint(bbox[0] + bbox[2])))
                        y2_list.append(
                            int(np.rint((bbox[1]+bbox[3]))))
                        category_list.append(anno_collect[j]['category_id']-1)

                anno = pd.DataFrame()
                anno['img_path'] = img_path_list
                anno['x1'] = x1_list
                anno['y1'] = y1_list
                anno['x2'] = x2_list
                anno['y2'] = y2_list
                anno['class'] = category_list
                anno['class'] = anno['class'].map(mapper)
            else:
                for j in range(anno_num):
                    if anno_collect[j]['image_id'] == img_id:
                        bbox = anno_collect[j]['bbox']
                        img_path_val_list.append(restrict_rele_path+img_name)
                        x1_val_list.append(int(np.rint(bbox[0])))
                        y1_val_list.append(int(np.rint(bbox[1])))
                        x2_val_list.append(
                            int(np.rint(bbox[0] + bbox[2])))
                        y2_val_list.append(
                            int(np.rint((bbox[1]+bbox[3]))))
                        category_val_list.append(
                            anno_collect[j]['category_id']-1)

                anno_val = pd.DataFrame()
                anno_val['img_path'] = img_path_val_list
                anno_val['x1'] = x1_val_list
                anno_val['y1'] = y1_val_list
                anno_val['x2'] = x2_val_list
                anno_val['y2'] = y2_val_list
                anno_val['class'] = category_val_list
                anno_val['class'] = anno_val['class'].map(mapper)

        anno.to_csv('/home/hywel/Documents/keras-retinanet/keras_retinanet/CSV/train_annotations.csv', index=None, header=None)
        anno_val.to_csv('/home/hywel/Documents/keras-retinanet/keras_retinanet/CSV/val_annotations.csv', index=None, header=None)
if __name__ == "__main__":
    restrict_image_info(label_path)
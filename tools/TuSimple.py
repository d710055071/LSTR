




import os
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import time
import random

def get_all_path(open_file_path,suffix):
    rootdir = open_file_path
    path_list = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        com_path = os.path.join(rootdir, list[i])
        if os.path.isfile(com_path) and os.path.splitext(com_path)[1] in suffix:
            path_list.append(com_path)
        if os.path.isdir(com_path):
            path_list.extend(get_all_path(com_path,suffix))
    return path_list

def getPointList(points_str):
    ret = []
    point_list = points_str.split(";")
    for onepoint in point_list:
        point = onepoint.split(",")
        ret.append([int(float(point[0])),int(float(point[1]))])
    return ret

label_map = {
    'fishline'          :0 + 1,         # 鱼骨线
    'variable_line'     :1 + 1,         # 可变车道
    'parking'           :2 + 1,         # 停车位线
    'SingleSolid'       :3 + 1,         # 单实线
    'SingleDotted'      :4 + 1,         # 单虚线
    'ForkSolid'         :5 + 1,         # 分歧线
    'Roadline'          :6 + 1,         # 路边
    'DoubleSolid'       :7 + 1,         # 双实线
    'DoubleDotted'      :8 + 1,         # 双虚线
    'SolidDotted'       :9 + 1,         # 左实右虚
    'DottedSolid'       :10 + 1,        # 左虚右实
    'Fence'             :11 + 1,        # 护栏
    'DoubleSingleSolid' :12 + 1,        # 双单实线
    'DoubleSingleDotted':13 + 1,        # 双单虚线
    'ShortDotted'       :14 + 1,        # 短虚线
    'Ignore'            :15 + 1,        # 忽略的线
}


def make_lable_data(annotations_path,save_img_path_str):
    ret = []
    # 解析标签文件
    tree = ET.parse(annotations_path)
    # 获取所有图片标签数据
    all_imgs_object = tree.findall('image')

    for oneimage_obj in tqdm(all_imgs_object):
        # 获取图片名称
        image_name = oneimage_obj.attrib['name']
        img_path = os.path.join(input_path,image_name)
        raw_file = os.path.join(save_img_path_str,image_name)
        # img = cv2.imread(img_path)
        img_h , img_w , img_c = 1080,1920,3
        h_samples = list(range(0, img_h, 10))

        # 绘制横线h_samples
        binary_image_h = np.zeros([img_h, img_w], np.uint8)

        for h in h_samples:
            cv2.line(binary_image_h,(0,h),(img_w-1,h),(255),thickness=1)
        lanes = []
        class_list = []
        # 获取车道标签数据
        all_lanes = oneimage_obj.findall('polyline')
        for onelane in all_lanes:
            
            binary_image = np.zeros([img_h, img_w], np.uint8)

            label_name = onelane.attrib['label']
            if label_name not in label_map:
                continue
            current_label = label_map[label_name]
            points_str = onelane.attrib['points']
            ret_point_list  = getPointList(points_str)
            cv2.polylines(binary_image,[np.array(ret_point_list)],False,(255),thickness=1) # 绘制车道线

            img_and = cv2.bitwise_and(binary_image,binary_image_h)

            single_lane = []
            
            for h in h_samples:
                start = False
                temp_w = []
                temp_w = np.where(img_and[h,:]>1)[0]
                if len(temp_w) > 0:
                    start = True
                if start:
                    half = len(temp_w) // 2
                    median = (temp_w[half] + temp_w[~half])/2
                    median = int(median)
                    # print("half:",half,median)
                    single_lane.append(median)
                else:
                    single_lane.append(-2)
            
            lanes.append(single_lane)
            class_list.append(current_label)
        if len(lanes) == 0 :
            continue
        dict_img_per = {"lanes":lanes,"h_samples":h_samples,"raw_file":raw_file,"categories":class_list}
        # json.dump(dict_img_per, file_obj)
        # file_obj.write('\n')
        ret.append(dict_img_per)
    return ret
if __name__ == "__main__":

    all_data_list = [
    '20230830',
    '20230904',
    '20230911',
    '20230919',
    '20230928_00',
    '20230928_01',
    '20230928_02',
    '20230928_03',
    '20230928_04',
    '20230928_05',
    '20230928_07',
    '20230928_09',
    '20230928_10',
    '20230928_11',
    '20230928_12',
    '20230928_13',
    '20230928_14',
    ]
    output_path = '/mnt/sda/mxdataset/mxlane_dataset/MxTuSimple/LaneDetection'
    input_path = "/mnt/sda/mxdataset/mxlane_dataset/MxTuSimple/LaneDetection/clips"
    train_annotations = "label_data_train.json"
    val_annotations = "label_data_val.json"

    train_file_obj = open(os.path.join(output_path,train_annotations),'w')
    val_file_obj = open(os.path.join(output_path,val_annotations),'w')

    
    all_labels = []
    for onepath in all_data_list:
        annotations_file_path = os.path.join(input_path,onepath,"annotations.xml")
        if onepath in ['20230830','20230904','20230911','20230919']:
            save_img_path_str = os.path.join("clips",onepath,"images/default")
        else:
            save_img_path_str = os.path.join("clips",onepath,"images")
        
        print("load ",annotations_file_path)
        all_labels.extend(make_lable_data(annotations_file_path,save_img_path_str))
    random.shuffle(all_labels)
    for onelabel in all_labels:
        if random.random()<0.99:
            json.dump(onelabel, train_file_obj)
            train_file_obj.write('\n')
        else:
            json.dump(onelabel, val_file_obj)
            val_file_obj.write('\n')
    train_file_obj.close()
    val_file_obj.close()

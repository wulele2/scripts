# 用于将RotateVOC的外接椭圆框标注转化
# 成RotateDetection中外接矩形框标注

import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
import xml.dom.minidom
import numpy as np
import os
import math
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# 类别索引字典
CLASSES_NAME = ( "__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant",  "sheep",   "sofa",  "train",  "tvmonitor",)
name2id=dict(zip(CLASSES_NAME,range(len(CLASSES_NAME))))
id2name=dict(zip(range(len(CLASSES_NAME)), CLASSES_NAME))


def forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)


def backward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([-1, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

            if theta == 0:
                w, h = h, w
                theta -= 90

            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([-1, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

            if theta == 0:
                w, h = h, w
                theta -= 90

            boxes.append([x, y, w, h, theta])
    return np.array(boxes, dtype=np.float32)

# 五点坐标-->八点坐标
# （这个api不太好用，还是用forward_convert更好，更具有鲁棒性）
# 函数仅能处理[-90,0]格式标注的物体。
def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box

def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []                                # 存储当前图像中所有物体的坐标和类别信息

    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        # if child_of_root.tag == 'size':
        #     for child_item in child_of_root:
        #         if child_item.tag == 'width':
        #             img_width = int(child_item.text)
        #         if child_item.tag == 'height':
        #             img_height = int(child_item.text)
        #
        # if child_of_root.tag == 'object':
        #     label = None
        #     for child_item in child_of_root:
        #         if child_item.tag == 'name':
        #             label = NAME_LABEL_MAP[child_item.text]
        #         if child_item.tag == 'bndbox':
        #             tmp_box = []
        #             for node in child_item:
        #                 tmp_box.append(float(node.text))
        #             assert label is not None, 'label is none, error'
        #             tmp_box.append(label)
        #             box_list.append(tmp_box)

        # ship
        # 获取图像的宽和高
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            # 遍历每个object
            tmp_box = [0., 0., 0., 0., 0.]
            for child_item in child_of_root:
                # 添加类别
                if child_item.tag == 'name':
                    label = name2id[child_item.text]
                # 添加bbox信息
                if child_item.tag == 'bndbox':
                    for node in child_item:
                        if node.tag == 'xmin':                   # 椭圆长轴
                            tmp_box[2] = float(node.text)
                        if node.tag == 'ymin':                   # 椭圆短轴
                            tmp_box[3] = float(node.text)
                        if node.tag == 'xmax':                   # 圆心
                            la,sa = node.text[1:-1].split(',')
                            tmp_box[0] = float(la)
                            tmp_box[1] = float(sa)
                        if node.tag == 'ymax':                   # 旋转角度
                            tmp_box[4] = float(node.text)
                    # 转成八点坐标
                    # tmp_box = coordinate_convert_r(tmp_box)
                    tmp_box.append(label)
                    box_list.append(tmp_box)
    gtbox_label = np.array(box_list, dtype=np.int32)
    gtbox_label = forward_convert(gtbox_label)                   # 五点 --> 八点
    return img_height, img_width, gtbox_label

def WriterXMLFiles(filename, path, gtbox_label_list, w, h, d):

    # dict_box[filename]=json_dict[filename]
    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    foldername = doc.createElement("folder")
    foldername.appendChild(doc.createTextNode("JPEGImages"))
    root.appendChild(foldername)

    nodeFilename = doc.createElement('filename')
    nodeFilename.appendChild(doc.createTextNode(filename))
    root.appendChild(nodeFilename)

    pathname = doc.createElement("path")
    pathname.appendChild(doc.createTextNode("xxxx"))
    root.appendChild(pathname)

    sourcename=doc.createElement("source")

    databasename = doc.createElement("database")
    databasename.appendChild(doc.createTextNode("Unknown"))
    sourcename.appendChild(databasename)

    annotationname = doc.createElement("annotation")
    annotationname.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(annotationname)

    imagename = doc.createElement("image")
    imagename.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(imagename)

    flickridname = doc.createElement("flickrid")
    flickridname.appendChild(doc.createTextNode("0"))
    sourcename.appendChild(flickridname)

    root.appendChild(sourcename)

    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodewidth.appendChild(doc.createTextNode(str(w)))
    nodesize.appendChild(nodewidth)
    nodeheight = doc.createElement('height')
    nodeheight.appendChild(doc.createTextNode(str(h)))
    nodesize.appendChild(nodeheight)
    nodedepth = doc.createElement('depth')
    nodedepth.appendChild(doc.createTextNode(str(d)))
    nodesize.appendChild(nodedepth)
    root.appendChild(nodesize)

    segname = doc.createElement("segmented")
    segname.appendChild(doc.createTextNode("0"))
    root.appendChild(segname)

    label_name_map = id2name

    for gtbox_label in gtbox_label_list:

        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(str(label_name_map[gtbox_label[-1]])))
        nodeobject.appendChild(nodename)

        nodetruncated = doc.createElement('truncated')
        nodetruncated.appendChild(doc.createTextNode(str(0)))
        nodeobject.appendChild(nodetruncated)

        nodedifficult = doc.createElement('difficult')
        nodedifficult.appendChild(doc.createTextNode(str(0)))
        nodeobject.appendChild(nodedifficult)

        nodepose = doc.createElement('pose')
        nodepose.appendChild(doc.createTextNode('xxx'))
        nodeobject.appendChild(nodepose)

        nodebndbox = doc.createElement('bndbox')
        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(gtbox_label[0])))
        nodebndbox.appendChild(nodex1)
        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(gtbox_label[1])))
        nodebndbox.appendChild(nodey1)
        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(gtbox_label[2])))
        nodebndbox.appendChild(nodex2)
        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(gtbox_label[3])))
        nodebndbox.appendChild(nodey2)
        nodex3 = doc.createElement('x3')
        nodex3.appendChild(doc.createTextNode(str(gtbox_label[4])))
        nodebndbox.appendChild(nodex3)
        nodey3 = doc.createElement('y3')
        nodey3.appendChild(doc.createTextNode(str(gtbox_label[5])))
        nodebndbox.appendChild(nodey3)
        nodex4 = doc.createElement('x4')
        nodex4.appendChild(doc.createTextNode(str(gtbox_label[6])))
        nodebndbox.appendChild(nodex4)
        nodey4 = doc.createElement('y4')
        nodey4.appendChild(doc.createTextNode(str(gtbox_label[7])))
        nodebndbox.appendChild(nodey4)

        # ang = doc.createElement('angle')
        # ang.appendChild(doc.createTextNode(str(angle)))
        # nodebndbox.appendChild(ang)
        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(os.path.join(path, filename), 'w')
    doc.writexml(fp, indent='\n')
    fp.close()

# 可视化五点坐标的图像
def draw5(filename, boxes, width =5, mode = 'xyxya'):

    img = Image.open(filename)
    w, h = img.size
    draw_obj = ImageDraw.Draw(img)

    for box in boxes:
        x_c, y_c, h, w, theta = box[0], box[1], box[2], box[3], box[4]
        rect = ((x_c, y_c), (h, w), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=(0, 0, 255),
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=(255, 0, 0),
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=(0, 0, 255),
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=(255, 0, 0),
                      width=width)
    # 显示图像
    plt.imshow(img)
    plt.show()
    #return img

# 可视化八点坐标的图像
def draw8(filename, boxes, width =5, mode = 'xyxyxyxy'):

    img = Image.open(filename)
    w, h = img.size
    draw_obj = ImageDraw.Draw(img)

    for box in boxes:
        box = np.int0(box)
        x1, y1, x2, y2, x3, y3, x4, y4 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]
        draw_obj.line(xy=[(x1, y1), (x2, y2) ],
                      fill=(0, 0, 255),
                      width=width)
        draw_obj.line(xy=[(x2, y2), (x3, y3)],
                      fill=(255, 0, 0),
                      width=width)
        draw_obj.line(xy=[(x3, y3), (x4, y4)],
                      fill=(0, 0, 255),
                      width=width)
        draw_obj.line(xy=[(x4, y4), (x1, y1)],
                      fill=(255, 0, 0),
                      width=width)
    # 显示图像
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':

    src_xml_path = '/home/wujian/data/RVOC2007/Ellipse_Annotations/' # ori ellipse path
    xml_path = '/home/wujian/data/RVOC2007/RRec_Annotations/'        # rec path

    img_path = '/home/wujian/data/RVOC2007/JPEGImages/'              # img_path
    save_path = '/home/wujian/data/RVOC2007/Visio/'                  # 可视化图像保存位置

    src_xmls = os.listdir(src_xml_path)
    for x in src_xmls:
        # 转成八点坐标
        x_path = os.path.join(src_xml_path, x)
        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(x_path)
        # 可视化八点坐标
        #img_name = img_path + x.split('.')[0] + '.jpg'
        #draw8(img_name, gtbox_label)

        WriterXMLFiles(x, xml_path, gtbox_label, img_width, img_height, 3)
        #wulele

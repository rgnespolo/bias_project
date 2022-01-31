import torch, os
import matplotlib.pyplot as plt
from detecto import core, utils, visualize
from torchvision import transforms
import xml.etree.cElementTree as et
import csv

metrics_file = open('log_rcnn_metrics.csv', 'a', newline='')
writer = csv.writer(metrics_file)

print('Checking CUDA - ' + str(torch.cuda.is_available()))

model = core.Model.load(
    r'cadis_light_and_dark_all_no_aug.pth', ['iris', 'pupil'])
# r'cadis_light_with_pupil.pth',[ 'iris', 'pupil'])


your_path = './CADIS/test_xml/'
files = filter(lambda x: x.endswith(('.png', '.jpg')), os.listdir(your_path))


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # BoxA  = predicted
    # boxB = expert selection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    precision = interArea / float(boxBArea)
    recall = interArea / float(boxAArea)
    # return the intersection over union value
    return iou, precision, recall



for file in files:
    file_path = your_path + file
    image = utils.read_image(file_path)
    predictions = model.predict_top(image)
    labels, boxes, scores = predictions


    # -------------------   extract all annotations from xml
    xml_file = file_path[:-4] + '.xml'
    xml_tree = et.parse(xml_file)
    root_xml = xml_tree.getroot()

    objects_list = []
    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []

    for name in root_xml.iter('name'):
        objects_list.append(name.text)
    for xmin in root_xml.iter('xmin'):
        xmin_list.append(xmin.text)
    for xmax in root_xml.iter('xmax'):
        xmax_list.append(xmax.text)
    for ymin in root_xml.iter('ymin'):
        ymin_list.append(ymin.text)
    for ymax in root_xml.iter('ymax'):
        ymax_list.append(ymax.text)
    print('objects extracted from xml: ' + str(objects_list))

    for i in objects_list:
        index_detected = labels.index(i)
        box = boxes[index_detected]
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        boxA = [x1, y1, x2, y2]
        index_annotated = objects_list.index(i)
        x1_xml = int(xmin_list[index_detected])
        x2_xml = int(xmax_list[index_detected])
        y1_xml = int(ymin_list[index_detected])
        y2_xml = int(ymax_list[index_detected])
        boxB = [x1_xml, y1_xml, x2_xml, y2_xml]

        iou, precision, recall = bb_intersection_over_union(boxA, boxB)
        print(i)
        print('iou: ' + str(iou) + ' precision: ' + str(precision) + ' recall: ' + str(recall))
        writer.writerow([file_path, i, iou,precision,recall, scores[index_detected].numpy()])
        #save to csv here - append

    visualize.show_labeled_image(image, boxes, labels)











    # top = torch.topk(predictions, 2)
    # predictions format: (labels, boxes, scores)

    # ['alien', 'bat', 'bat']

    #           xmin       ymin       xmax       ymax
    # tensor([[ 569.2125,  203.6702, 1003.4383,  658.1044],
    #         [ 276.2478,  144.0074,  579.6044,  508.7444],
    #         [ 277.2929,  162.6719,  627.9399,  511.9841]])

    # tensor([0.9952, 0.9837, 0.5153])
    # labels = labels[0:1]
    # boxes = torch.topk(boxes, 2)
    # scores = torch.topk(scores, 2)
    # print(labels)
    # print(boxes)
    # print(scores)
import torch, os
import matplotlib.pyplot as plt
from detecto import core, utils, visualize
from torchvision import transforms
print('Checking CUDA - ' + str(torch.cuda.is_available()))



model = core.Model.load(
            r'cadis_light_and_dark_all_no_aug.pth', ['iris', 'pupil'])
            #r'cadis_light_with_pupil.pth',[ 'iris', 'pupil'])


your_path = './CADIS/test_frames_ngenuity/'
files = filter(lambda x: x.endswith(('.png','.jpg')), os.listdir(your_path))

print(files)


for file in files:
    # Specify the path to your image
    file_path = your_path + file
    print(file_path)
    #image = cv2.imread(file_path)
    image = utils.read_image(file_path)
    #image = utils.read_image('phacoemulsification_quadrant_removal_Default_Doctor_no_date 110037 AM-convertedmp4_4437-01.png')
    predictions = model.predict_top(image)
    print(predictions)
    #top = torch.topk(predictions, 2)
    # predictions format: (labels, boxes, scores)
    labels, boxes, scores = predictions

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
    visualize.show_labeled_image(image, boxes, labels)

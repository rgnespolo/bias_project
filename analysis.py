import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from skimage.transform import resize
import os


# Load the trained model
model = torch.load('./CADIS/iris weights/CADIS_weights_iris_good_epoch_2_lr5.pt')
# Set the model to evaluate mode
model.eval()

# Read the log file using pandas into a dataframe
df = pd.read_csv('./CADIS/log.csv')

# Plot all the values with respect to the epochs
df.plot(x='epoch', figsize=(15, 8));
# plt.show()
print(df[['Train_auroc', 'Test_auroc']].max())

ino = 2
# Read  a sample image and mask from the data-set
# crack
# img = cv2.imread(f'./CrackForest/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,480)
# mask = cv2.imread(f'./CrackForest/Masks/{ino:03d}_label.PNG')
# pupil
#img = cv2.imread(f'./CADIS/Images/Video1_frame000090.png')

your_path = './CADIS/bias_test_light'
files = os.listdir(your_path)
print(files)

for file in files:
    file_path = f'./CADIS/bias_test_light/'+file
    print(file_path)
    img = cv2.imread(file_path)
    img_orig = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).reshape(1, 3, 256, 256)
    #mask = cv2.imread(f'./CADIS/Masks_iris/Video1_frame000090.png')
    # prediction
    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor) / 255)

    # Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
    # plt.hist(a['out'].data.cpu().numpy().flatten())

    model_output = (a['out'].cpu().detach().numpy()[0][0] > 0.5)
    model_output = model_output.astype(np.uint8)  # convert to an unsigned byte
    model_output *= 255
    model_output = cv2.cvtColor(model_output, cv2.COLOR_GRAY2BGR)
    merged_figure = cv2.addWeighted(img_orig, 1, model_output, 0.1,0)
    merged_figure = cv2.resize(merged_figure, (int(1280/2), int(720/2)))
    cv2.imshow('final', merged_figure)
    cv2.waitKey(0)











# # Plot the input image, ground truth and the predicted output
# plt.figure(figsize=(10, 10))
# plt.subplot(131)
# plt.imshow(img[0, ...].transpose(1, 2, 0), aspect=0.5625)  # transpose bgr to rgb, aspect ratio 16:9
# plt.title('Image')
# plt.axis('off')
#
# plt.subplot(132)
# plt.imshow(model_output, aspect=0.5625)
# plt.title('Segmentation Output')
# plt.axis('off')
# plt.savefig('./CADIS/SegmentationOutput.png', bbox_inches='tight')
# plt.show()

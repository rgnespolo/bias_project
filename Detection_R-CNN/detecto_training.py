import torch
import matplotlib.pyplot as plt
from detecto import core, utils
from torchvision import transforms
print('Checking CUDA - ' + str(torch.cuda.is_available()))

#being sure that png 4 channels are transformed to 3 channels
augmentations = transforms.Compose([
    transforms.Lambda(lambda x: x[:,:,:3]),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

#set dataset folder - train and validation
dataset = core.Dataset(r'C:\Users\MGS\Documents\GitHub\bias_project\Detection_R-CNN\CADIS\all_no_aug\train', transform=augmentations)
val_dataset = core.Dataset(r'C:\Users\MGS\Documents\GitHub\bias_project\Detection_R-CNN\CADIS\all_no_aug\valid')

loader = core.DataLoader(dataset, batch_size=2, shuffle=True)

#set the labels available
model = core.Model(['iris','pupil'])

#fit the model
losses = model.fit(loader, val_dataset, epochs=100, learning_rate=0.005,
                   lr_step_size=5, verbose=True)


#plot the losses
plt.plot(losses)
plt.show()


model.save('cadis_light_and_dark_all_no_aug.pth')
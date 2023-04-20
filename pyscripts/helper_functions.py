import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import torch
import pickle



def load_image(path, IMG_SZE = 4000, save_name = str):
    # Load images into a list with save name.
    save_name = os.path.join(path[-11:-4]+ ".csv")
    img_array = cv2.imread(path)
    return save_name, cv2.resize(img_array, (IMG_SZE, IMG_SZE))

def load_model(yolo_path, weight_path, confidence = 0.75):
    # Loads a YOLO model with custom weights
    model = torch.hub.load(yolo_path, 'custom', path=weight_path, source='local')  # local repo
    model.conf = confidence
    return model

def plot_conf(df = pd.DataFrame(), C = float, img = np.zeros((10, 10)), fig = plt.figure(), idx = int):
    for ixcx, i in df.iterrows():
        if i['name'] == 'ic':
            x, w = dict(i)['xmin'], dict(i)['xmax'] - dict(i)['xmin']
            y, h = dict(i)['ymin'], dict(i)['ymax'] - dict(i)['ymin']
            plt.subplot(3, 3, idx + 1)
            rectangle = plt.Rectangle((x,y), w, h, facecolor="none",ec="red")
            fig.gca().add_patch(rectangle)
            plt.imshow(img)
    plt.title(f'Confidence  = {C:.2f}')
    #plt.axis('off')
    plt.ylim(600, 3000)
    plt.xlim(800, 3600)
    return


def to_cpu(image):
    return image.detach().cpu().numpy().transpose(1, 2, 0)

def plot_recon_mse(in1, out1):
    images = [in1, out1]
    titles = ["original", "Reconstructed", "Error"]
    fig = plt.figure()
    for i in range(3):
        if i < 2:
            fig.add_subplot(1, 3, i +1)
            plt.imshow(images[i])
            plt.axis('off')
            plt.title(titles[i])
        else:
            fig.add_subplot(1, 3, i +1)
            plt.imshow(((in1 - out1) **2).mean(axis = 2), cmap = 'jet', vmin = 0, vmax = 0.01)
            plt.axis('off')
            plt.title(titles[i])
    return 


def comps_mse(comp_list = list, device = str()):
    # Prepare data for CNN autoencoder
    imgs = torch.Tensor((np.array(comp_list)/255).transpose(0, 3, 1, 2)).to(device)
    testloader = DataLoader(imgs, batch_size=1)

    # Reconstruct items and find.
    mse_all = []
    for idx, i in enumerate(imgs):
        mse_all.append(((to_cpu(i) - to_cpu(CNN_AE(i))) **2).mean())
    return mse_all

def plot_PCBwithDetection(df_ic = pd.DataFrame(), save_name = str(), mse_all = list, threshold = float, img = np.zeros((10, 10)), printimg = True):
    fig = plt.figure(figsize = (15, 10))
    for idx, i in df_ic.reset_index().iterrows():
        x, w = dict(i)['xmin'], dict(i)['xmax'] - dict(i)['xmin']
        y, h = dict(i)['ymin'], dict(i)['ymax'] - dict(i)['ymin']
        if mse_all[idx-1] < threshold:
            rectangle = plt.Rectangle((x,y), w, h, facecolor="none",ec="blue")
            fig.gca().add_patch(rectangle)
        else:
            rectangle = plt.Rectangle((x,y), w, h, facecolor="red",ec="red", alpha = 0.3)
            fig.gca().add_patch(rectangle)
        if printimg:
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join('../processed_images', save_name[:-4] + ".jpg"))

    return

    
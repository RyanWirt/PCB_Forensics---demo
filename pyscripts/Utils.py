
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import torch
import pickle
import numpy as np


def load_image(path: str, img_size: int = 4000, save_name: str = None) -> tuple:

    """
    Load image from file path and resize it to specified dimensions. Optionally, specify a name for the file to save
    as. Returns a tuple containing the file name and the resized image.

    :param path: A string containing the file path to the image file.
    :param img_size: An integer specifying the desired size for the output image (default: 4000).
    :param save_name: An optional string specifying the name for the output file (default: None).
    :return: A tuple containing the file name and the resized image array.
    """

    # If no save_name is specified, create a default name using the last 8 characters of the path.
    if save_name is None:
        save_name = os.path.join(path[-10:-4] + ".csv")

    # Load the image array using OpenCV imread function.
    img_array = cv2.imread(path)

    # Resize the image array to specified dimensions using OpenCV resize function.
    resized_img = cv2.resize(img_array, (img_size, img_size))

    # Return a tuple containing the file name and the resized image array.
    return save_name, resized_img


def load_model(yolo_path: str, weight_path: str, confidence: float = 0.75) -> torch.nn.Module:
    """
    Loads a YOLO model with custom weights and minimum confidence threshold.

    :param yolo_path: A string containing the local path to the YOLO repository.
    :param weight_path: A string containing the local path to the custom weight file.
    :param confidence: A float specifying the minimum confidence threshold for object detection (default: 0.75).
    :return: A PyTorch neural network module representing the loaded YOLO model.
    """

    # Load the YOLO model from the specified repository and custom weights file.
    model = torch.hub.load(yolo_path, 'custom', path=weight_path, source='local')

    # Set the minimum confidence threshold for object detection.
    model.conf = confidence

    # Return the loaded YOLO model.
    return model

def plot_conf(df: pd.DataFrame, C: float, img: np.ndarray, fig: plt.Figure, idx: int) -> None:
    """
    Plots a rectangle on an image at the location specified by a Pandas DataFrame.

    :param df: A Pandas DataFrame containing the location information for the rectangle.
    :param C: A float representing the confidence level.
    :param img: A NumPy array representing the image.
    :param fig: A Matplotlib Figure object.
    :param idx: An integer representing the index of the subplot.
    :return: None
    """
    
    # Loop through each row of the DataFrame.
    for ixcx, i in df.iterrows():
        # Check if the name of the object is 'ic'.
        if i['name'] == 'ic':
            # Get the x, y, width, and height values from the DataFrame.
            x, w = dict(i)['xmin'], dict(i)['xmax'] - dict(i)['xmin']
            y, h = dict(i)['ymin'], dict(i)['ymax'] - dict(i)['ymin']
            
            # Plot a rectangle on the image at the specified location.
            plt.subplot(3, 3, idx + 1)
            rectangle = plt.Rectangle((x,y), w, h, facecolor="none",ec="red")
            fig.gca().add_patch(rectangle)
            plt.imshow(img)
    
    # Set the plot title and axis limits.
    plt.title(f'Confidence = {C:.2f}')
    plt.ylim(600, 3000)
    plt.xlim(800, 3600)
    
    # Return None since the function only plots the rectangle and doesn't return any values.
    return None

def to_cpu(image):
    return image.detach().cpu().numpy().transpose(1, 2, 0)

def plot_recon_mse(in1: np.ndarray, out1: np.ndarray) -> None:
    """
    Plots the original image, reconstructed image, and the corresponding mean squared error.

    :param in1: A NumPy array representing the original image.
    :param out1: A NumPy array representing the reconstructed image.
    :return: None
    """
    
    # Create a list of the input and output images.
    images = [in1, out1]
    
    # Create a list of titles for each plot.
    titles = ["Original", "Reconstructed", "Error"]
    
    # Create a figure object with 3 subplots.
    fig = plt.figure()
    
    # Loop through each plot.
    for i in range(3):
        if i < 2:
            # Add a subplot for the input or output image.
            fig.add_subplot(1, 3, i + 1)
            plt.imshow(images[i])
            plt.axis('off')
            plt.title(titles[i])
        else:
            # Add a subplot for the mean squared error between the input and output images.
            fig.add_subplot(1, 3, i + 1)
            mse = ((in1 - out1) ** 2).mean(axis=2)
            plt.imshow(mse, cmap='jet', vmin=0, vmax=0.01)
            plt.axis('off')
            plt.title(titles[i])
    
    # Return None since the function only plots the images and doesn't return any values.
    return None

def comps_mse(comp_list: list, device: str = 'cpu') -> list:
    """
    Calculates the mean squared error (MSE) between the input and reconstructed images for a list of compressed images.

    :param comp_list: A list of compressed images.
    :param device: A string specifying the device to use for computations (e.g. 'cpu' or 'cuda').
    :return: A list of MSE values for each compressed image.
    """
    
    # Convert the compressed images to a tensor and move them to the specified device.
    imgs = torch.Tensor((np.array(comp_list)/255).transpose(0, 3, 1, 2)).to(device)
    
    # Create a DataLoader object for the input images.
    testloader = DataLoader(imgs, batch_size=1)
    
    # Reconstruct each item and calculate the MSE.
    mse_all = []
    for idx, i in enumerate(imgs):
        # Reconstruct the input image using the CNN autoencoder and calculate the MSE.
        reconstructed = CNN_AE(i)
        mse = ((to_cpu(i) - to_cpu(reconstructed)) ** 2).mean()
        
        # Add the MSE value to the list of MSE values.
        mse_all.append(mse)
    
    # Return the list of MSE values.
    return mse_all

def plot_PCBwithDetection(df_ic=pd.DataFrame(), save_name=str(), mse_all=list, threshold=float, img=np.zeros((10, 10)), printimg=True):
    """
    Plots PCB image with object detection bounding boxes overlayed. 
    Bounding boxes are colored red if the MSE of the corresponding image patch 
    is above the threshold, and blue otherwise.

    Parameters:
    -----------
    df_ic : pandas DataFrame
        Dataframe containing object detection information (i.e., xmin, xmax, ymin, ymax).
    save_name : str
        File name to save the resulting image.
    mse_all : list
        List of mean squared errors between the original and reconstructed image patches.
    threshold : float
        MSE threshold above which bounding boxes will be colored red.
    img : np.array
        Image array to be plotted.
    printimg : bool
        Whether or not to save the plotted image.

    Returns:
    --------
    None.
    """
    fig = plt.figure(figsize=(15, 10))
    for idx, i in df_ic.reset_index().iterrows():
        x, w = dict(i)['xmin'], dict(i)['xmax'] - dict(i)['xmin']
        y, h = dict(i)['ymin'], dict(i)['ymax'] - dict(i)['ymin']
        if mse_all[idx-1] < threshold:
            rectangle = plt.Rectangle((x,y), w, h, facecolor="none",ec="blue")
            fig.gca().add_patch(rectangle)
        else:
            rectangle = plt.Rectangle((x,y), w, h, facecolor="red",ec="red", alpha=0.3)
            fig.gca().add_patch(rectangle)

    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join('../processed_images', save_name[:-4] + ".jpg"))
    return

def load_cnn_autoencoder(model_path):
    # Set the device to CPU

    if torch.cuda.is_available():   
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    # Load the CNN autoencoder model from the given file path
    CNN_AE = torch.load(model_path)
    
    # Move the model to the specified device
    CNN_AE.to(device)
    
    # Set the model to evaluation mode
    CNN_AE.eval()
    
    # Return the loaded model
    return device, CNN_AE
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Detect ICs on PCB classify them as "normal" or "anomalous"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




from Utils import *
from CNN_autoencoder import *

def main(confidence = 0.5, threshold = 0.05, img_folder = "../demo-data", 
         weight_path = "../YoloPCB/training_results/PCB_detection_best.pt", 
         yolo_path = "../YoloPCB", model_path = '../models/IC_AE_cpu.pth'):

    """
    Detect IC components on PCB images using YOLOv5 object detection model and CNN autoencoder.

    Args:
    confidence (float): confidence threshold for object detection using YOLOv5 (default: 0.5)
    threshold (float): threshold value for anomaly detection using CNN autoencoder (default: 0.05)
    img_folder (str): folder path containing PCB images to be processed (default: "../demo-data")
    weight_path (str): path to YOLOv5 model weights (default: "../YoloPCB/training_results/PCB_detection_best.pt")
    yolo_path (str): path to YOLOv5 codebase (default: "../YoloPCB")
    model_path (str): path to trained CNN autoencoder model (default: "../models/IC_AE_cpu.pth")
    """
    # Load images that you want to detect
    image_list = [load_image(os.path.join(img_folder, img)) for img in os.listdir(img_folder)]

    # Load CNN autoencoder model for anomaly detetection
    device, CNN_AE = load_cnn_autoencoder(model_path)

    # Load YOLOv5 object detection model
    model = load_model(yolo_path, weight_path, confidence = confidence)
    # Create a dir to save processed images
    try:
        os.mkdir('../processed_images')
    except:
        print('Directory already exists')   
    
    # Loop through each image in the folder
    for idx, (save_name, img_array1) in enumerate(image_list):
        
        # Detect objects using YOLOv5
        results = model(img_array1) # batch of images

        # Convert results to pandas dataframe
        df  = pd.DataFrame(results.pandas().xyxy[0])
        df[['xmin', 'xmax','ymin', 'ymax']] = df[['xmin', 'xmax','ymin', 'ymax']].astype(int)
        df_ic = df[df['name'] ==  'ic']

        # Extract IC components from the image and process them using CNN autoencoder
        try:
            ic_s = [cv2.resize(img_array1[dict(i)['ymin']:dict(i)['ymax'], dict(i)['xmin']:dict(i)['xmax']], (400, 400)) for ixc, i in df_ic.iterrows()]
            
            # Prepare images for CNN detection
            imgs = torch.Tensor((np.array(ic_s)/255).transpose(0, 3, 1, 2)).to(device)
            
            # Calculate error for encoded images.
            mse_all = [(((to_cpu(i) - to_cpu(CNN_AE(i)))**2).mean()) for idx, i in enumerate(imgs)]

            # Plot the original image with detected IC components and anomalies highlighted
            plot_PCBwithDetection(df_ic = df_ic, save_name = save_name, mse_all = mse_all, threshold = threshold, img = img_array1, printimg = True)
            print(imgs.shape)

        except:
            # if no detected IC compoenets print image name and save image
            print(save_name[:-4] + ".jpg")
            cv2.imwrite(os.path.join('../processed_images', save_name[:-4] + ".jpg"), img_array1)

if __name__ == '__main__':
    main()
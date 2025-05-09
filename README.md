These codes have been created to be used in Google Colab, some modifications may need to be made.
Both, original dataset and data augmentation images can be validated.
Grad-CAM takes the last layer before classification from the SE DenseNET and allows the visualization of the most relevant features in each image.
Gray denoising was the best data augmentation technique, but flipped and noisy had a great performance too. 
Rotated, grey, cropped and perspective showed similar results to the original dataset.
Color denoising and scaled showed much worse performance than the original dataset and should not be taken into consideration in further trials.

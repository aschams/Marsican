import cv2
import numpy as np

def augment_img(img:str, filepath:str, k:list=[0], lr:bool=True, periods:int=3, slashes:int=4):
    """
    Inputs:
        img: str; containing filepath to the image being augmented
        filepath: string containing filepath to save the augmented images to.
        k: list; integers of the numbers of rotations of each image produced.
            ex: function will produce len(k) images. [0, 1, 3] will produce 1 image with no rotation,
                1 image with 1 90 degree rotation, and 1 image with 3 90 degree (or 270) degree rotation.
        lr: bool; Whether or not to flip the image left-right.
        periods: int; number of periods in img filepath. Specifies which segment to find the filename in.
        slashes: int; number of slashes in img filepath. Used to find the filename.
    Returns: None
    """
    filename = img.split('.')[-2].split('/')[-1]
    img_arr = cv2.imread(img)
    img_arr = cv2.resize(img_arr, (128, 128))
    if lr:
        for rot in k:
            flip_lr = np.fliplr(img_arr)
            flip_lr_rot = np.rot90(flip_lr, rot, axes=(0,1))
            cv2.imwrite(f'{filepath}/{filename}_rot{rot}_{1}.png', flip_lr_rot)
    else:
        for rot in k:
            flip_lr_rot = np.rot90(img_arr, rot, axes=(0,1))
            cv2.imwrite(f'{filepath}/{filename}_rot{rot}_{0}.png', flip_lr_rot)

def augment_img_both(img:str, filepath:str, k:list=[0], periods:int=3, slashes:int=4):
    """
    Performs data augmentation on an image both with and without a left-right flip.
    Inputs:
        img: ndarray of pixel values for an image
        filepath: string containing filepath to save the augmented images to.
        k: list; integers of the numbers of rotations of each image produced.
            ex: function will produce len(k) images. [0, 1, 3] will produce 1 image with no rotation,
                1 image with 1 90 degree rotation, and 1 image with 3 90 degree (or 270) degree rotation.
    Returns: None
    """
    augment_img(img, filepath, k, periods, slashes)
    augment_img(img, filepath, k, lr=False, periods=periods, slashes=slashes)

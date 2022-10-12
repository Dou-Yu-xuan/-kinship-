import cv2
import numpy as np

def readDepthFile(filepath, image_size=(480,640)):
    depth = np.fromfile(filepath, dtype='float16')  # read as half-float
    
    # sometimes data has additional padding, so we need to remove it
    if (depth.shape[0] > image_size[0]*image_size[1]) :         
        paddingSize = image_size[0]*image_size[1] - depth.shape[0]
        depth = depth[:paddingSize]
    
    depth = depth.reshape(image_size)               # rearrange to iPhoneX portrait resolution

    return depth                                    # depth is in M

def normalizeGrayscale(arr):
    arr[np.isnan(arr)] = 0                          # converting nan to zero for visualization
    arr = arr - arr.min()
    arr = arr / (arr.max() - arr.min())

    return arr


if __name__ == "__main__":
    depthArr = readDepthFile('/home/wei/Documents/0.dpt')
    normalizedDepthArr = normalizeGrayscale(depthArr).astype(float)

    cv2.imshow('depth data normalized', normalizedDepthArr)
    # cv2.imwrite('/home/wei/Documents/0.jpg',normalizedDepthArr)
    print('Press Esc to Exit ... ')
    cv2.waitKey(-1)
# image_masker.py

import cv2
import numpy as np

def resize_image(image_path, scale_x=0.4, scale_y=0.4):
    """
    Resizes the image.

    :param image_path: Path to the image file.
    :param scale_x: Horizontal scale factor.
    :param scale_y: Vertical scale factor.
    :return: Resized image.
    """
    image = cv2.imread(image_path)
    return cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)

def select_roi(image):
    """
    Allows user to select a Region of Interest (ROI) on the image.

    :param image: Image on which to select the ROI.
    :return: Coordinates of the selected ROI.
    """
    return cv2.selectROI('Select ROI', image, True)

def create_mask(image, roi, inverse=True):
    """
    Creates a binary mask for the image based on the selected ROI. 
    The mask will be strictly black and white.

    :param image: Source image.
    :param roi: Region of Interest (x, y, width, height).
    :param inverse: If True, the area outside the ROI is white. Defaults to False.
    :return: Binary mask image.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if inverse:
        # Inverse mask: the area outside the ROI is white
        mask[:] = 255  # Set the whole mask to white
        mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 0  # Set ROI to black
    else:
        # Regular mask: the area inside the ROI is white
        mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 255  # Set ROI to white

    return mask

    return cv2.bitwise_and(image, image, mask=mask2)



def main(image_path):
    """
    Main function to process the image.

    :param image_path: Path to the image file.
    """
    src = resize_image(image_path)
    r = select_roi(src)
    result = create_mask(src, r)
    cv2.imshow("Masked Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])  # Pass image path as command line argument

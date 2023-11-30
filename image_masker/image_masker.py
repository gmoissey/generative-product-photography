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

def create_mask(image, roi):
    """
    Creates a mask for the image based on the selected ROI using GrabCut algorithm.

    :param image: Source image.
    :param roi: Region of Interest.
    :return: Masked image.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, roi, bgdmodel, fgdmodel, 10, mode=cv2.GC_INIT_WITH_RECT)

    # Create a binary mask where the background is white (255) and the foreground (object) is black (0)
    mask2 = np.where((mask == 2) | (mask == 0), 255, 0).astype('uint8')

    return mask2

def main(image_path):
    """
    Main function to process the image.

    :param image_path: Path to the image file.
    """
    src = resize_image(image_path)
    r = select_roi(src)
    mask = create_mask(src, r)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])  # Pass image path as command line argument

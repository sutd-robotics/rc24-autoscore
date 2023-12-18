import logging
from typing import Tuple

import cv2
from cv2.typing import MatLike
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Define constants
H_MAX = 179
S_MAX = V_MAX = 255


# Define labels
HMIN_LABEL = 'HMin'
SMIN_LABEL = 'SMin'
VMIN_LABEL = 'VMin'
HMAX_LABEL = 'HMax'
SMAX_LABEL = 'SMax'
VMAX_LABEL = 'VMax'
TITLE =  'image'


def init() -> None:
    """Initialises the output window.

    See Also
    --------
    https://stackoverflow.com/a/58194879
    """

    # Create callable function
    nothing = lambda *_: None

    # Create GUI overlay
    cv2.namedWindow(TITLE)
    cv2.createTrackbar(HMIN_LABEL, TITLE, 0, H_MAX, nothing)
    cv2.createTrackbar(SMIN_LABEL, TITLE, 0, S_MAX, nothing)
    cv2.createTrackbar(VMIN_LABEL, TITLE, 0, V_MAX, nothing)
    cv2.createTrackbar(HMAX_LABEL, TITLE, 0, H_MAX, nothing)
    cv2.createTrackbar(SMAX_LABEL, TITLE, 0, S_MAX, nothing)
    cv2.createTrackbar(VMAX_LABEL, TITLE, 0, V_MAX, nothing)

    # Set default value for max HSV trackbars
    cv2.setTrackbarPos(HMAX_LABEL, TITLE, H_MAX)
    cv2.setTrackbarPos(SMAX_LABEL, TITLE, S_MAX)
    cv2.setTrackbarPos(VMAX_LABEL, TITLE, V_MAX)


def update() -> Tuple[np.ndarray, np.ndarray]:
    """Updates minimum and maximum HSV detection range based on trackbar positions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two 1D `np.ndarray`s consisting of three integers from 0-255,
        each representing the values of H, S, and V respectively,
        where the first `np.ndarray` represents the lower bound and
        the second `np.ndarray` represents the upper bound.

    See Also
    --------
    https://stackoverflow.com/a/58194879
    """

    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos(HMIN_LABEL, TITLE)
    sMin = cv2.getTrackbarPos(SMIN_LABEL, TITLE)
    vMin = cv2.getTrackbarPos(VMIN_LABEL, TITLE)
    hMax = cv2.getTrackbarPos(HMAX_LABEL, TITLE)
    sMax = cv2.getTrackbarPos(SMAX_LABEL, TITLE)
    vMax = cv2.getTrackbarPos(VMAX_LABEL, TITLE)

    # Set minimum and maximum HSV values to display
    return np.array([hMin, sMin, vMin]), np.array([hMax, sMax, vMax])


def segment(image: MatLike) -> None:
    """Perform colour segmentation on a given image.

    The output of the segmentation is automatically updates on the output window.

    Parameters
    ----------
    image : MatLike
        The image to perform segmentation on.

    See Also
    --------
    https://stackoverflow.com/a/58194879
    """

    # Obtain lower and upper bounds for segmentation
    lower, upper = update()

    # Create HSV Image and threshold into a range
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # Display output image
    cv2.imshow(TITLE, output)


if __name__ == '__main__':
    logger.info('Initialising...')
    init()
    # cap = cv2.VideoCapture(0)
    logger.info('Ready.')

    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning('Unable to receive frame.')
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        segment(frame)

    logger.info('Exiting...')
    cap.release()
    """

    image = cv2.imread('assets/sample.png')
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        segment(image)
    logger.info('Exiting...')

    cv2.destroyAllWindows()

import logging
import socket
from typing import Tuple

from tune import (
    H_MAX, S_MAX, V_MAX,
    TUNE_RED, TUNE_BLUE,
    segment
)

import cv2
from cv2.typing import MatLike
import numpy as np


# Define debugging
VERBOSE = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG if VERBOSE else logging.INFO)


# Define constants
ROCK_AREA_THRESHOLD = 2000
HOST = '192.168.1.120'
PORT = 9998


# Define corners
TOP_LEFT_CORNER, TOP_RIGHT_CORNER, BOTTOM_LEFT_CORNER, BOTTOM_RIGHT_CORNER = range(4)
CAM_CORNER = TOP_LEFT_CORNER


def getBound(
        bgr: np.ndarray,
        var: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the rough lower and upper bounds for detecting a given colour.

    Parameters
    ----------
    bgr : np.ndarray
        A 1D `np.ndarray` consisting of three integers from 0-255,
        representing the given colour by its B, G, and R values respectively.
    var : int, default=50
        The variance in the minimum/maximum hue value from the hue of the given colour.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two 1D `np.ndarray`s consisting of three integers from 0-255,
        each representing the values of H, S, and V respectively,
        where the first `np.ndarray` represents the lower bound and
        the second `np.ndarray` represents the upper bound.
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    logger.debug('HSV for %s is %s', bgr, hsv)

    h, _, _ = hsv[0][0]
    return np.array([max(h - var, 0), 100, 100]), np.array([min(h + var, H_MAX), S_MAX, V_MAX])


def watershed(mask: MatLike) -> Tuple[MatLike, MatLike, MatLike]:
    """Perform watershed algorithm to detect connected contour components.

    Parameters
    ----------
    mask : MatLike
        The colour segmentation mask of the given image.

    Returns
    -------
    Tuple[MatLike, MatLike, MatLike]
        The markers of each pixel,
        followed by the statistics of each of the markers,
        followed by the centroids of each of the markers.

    See Also
    --------
    https://stackoverflow.com/a/46084597
    https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
    https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga5ed7784614678adccb699c70fb841075
    """

    # Perform Gaussian blur on mask
    #blur = cv2.GaussianBlur(mask, (7, 7), 5)
    blur = cv2.medianBlur(mask, 5)

    # Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

    # Binarize gradient
    lowerb = np.array([0, 0, 0])
    upperb = np.array([25, 25, 25])
    binary = cv2.inRange(gradient, lowerb, upperb)

    # Flood fill from the edges
    for row in range(HEIGHT):
        if binary[row, 0] == 255:
            cv2.floodFill(binary, None, (0, row), 0)
        if binary[row, WIDTH - 1] == 255:
            cv2.floodFill(binary, None, (WIDTH - 1, row), 0)
    for col in range(WIDTH):
        if binary[0, col] == 255:
            cv2.floodFill(binary, None, (col, 0), 0)
        if binary[HEIGHT - 1, col] == 255:
            cv2.floodFill(binary, None, (col, HEIGHT - 1), 0)

    # Cleaning up mask
    foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # Creating background and unknown mask for labeling
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    background = cv2.dilate(foreground, kernel, iterations=3)
    unknown = cv2.subtract(background, foreground)

    # Perform watershed algorithm and get all stats information
    # Mark unknown regions as -1 and background regions as 1
    _, markers, stats, centroids = \
        cv2.connectedComponentsWithStatsWithAlgorithm(foreground, 8, cv2.CV_32S, cv2.CCL_DEFAULT)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(mask, markers)

    return markers, stats, centroids


def analyse(
        markers: MatLike,
        stats: MatLike,
        centroids: MatLike,
        area_threshold: int
) -> MatLike:
    """Count number of detected rocks in image.

    The detected components are evaluated based on two criteria:
    - Area: The area of the component must meet the threshold.

    If a given component does not satisfy any of the criteria, the detection is nullified.

    Parameters
    ----------
    markers : MatLike
        The markers of each pixel.
    stats : MatLike
        The statistics of each detected component.
    centroids : MatLike
        The centroid of each detected component.
    area_threshold : int
        The minimum area of valid components.

    Returns
    -------
    MatLike
        The markers of each pixel after filtering.
    """

    for marker, stat, centroid in zip(np.unique(markers)[1:], stats, centroids):
        # Obtain relevant statistics
        x, y = map(int, centroid)
        area = stat[cv2.CC_STAT_AREA]

        # Perform filtering
        if area < area_threshold:
            markers[markers == marker] = -1

    return markers


def show(
        image: MatLike,
        markers: MatLike,
        centroids: MatLike,
        title: str
) -> None:
    """Display the detected components on the original image.

    Parameters
    ----------
    image : MatLike
        The original image.
    markers : MatLike
        The markers of each pixel.
    centroids : MatLike
        The centroids of each of the detected components.
    title : str
        The title of the resultant window.
    """

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # Assign the markers a hue between 0 and 179 (max H value)
    hue_markers = np.uint8(H_MAX * np.float32(markers) / np.max(markers))
    blank_channel = S_MAX * np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    marker_img = cv2.merge([hue_markers, blank_channel, blank_channel])
    marker_img = cv2.cvtColor(marker_img, cv2.COLOR_HSV2BGR)

    # Label the original image with the watershed markers
    labelled_img = image.copy()
    labelled_img[markers > 1] = marker_img[markers > 1]  # 1 is background color
    labelled_img = cv2.addWeighted(image, 0.5, labelled_img, 0.5, 0)

    # Label the centroid(s)
    for marker in np.unique(markers)[1:]:
        if marker == 1:
            continue

        x, y = map(int, centroids[marker - 1])
        cv2.circle(
            labelled_img, (x, y),
            15, (255, 255, 255), 2
        )

def find(
        image: MatLike,
        colour: int,
        *,
        verbose: bool = False
) -> int:
    """Pipeline for finding rocks of a specified colour.

    Colour detection follows the following pipeline:
    Colour-based segmentation > Flood-fill and Watershed Algorithms > Connected Component Analysis

    Parameters
    ----------
    image : MatLike
        The original image to find rock game elements from.
    colour : int
        The colour to find.
        Expecting one of `TUNE_RED`, `TUNE_BLUE`, or `TUNE_WHITE`.
    verbose : bool, default=False
        True to output the intermediate result(s), False otherwise.

    Returns
    -------
    int
        The number of rocks detected.

    See Also
    --------
    `tune.segment(image, lower, upper, *, invert, hsl, title)`
    `watershed(mask)`
    `analyse(markers, stats, centroids, area_threshold, region)`
    """

    # Perform sanity checking
    if not colour in (TUNE_RED, TUNE_BLUE):
        logger.error('colour=%i should be one of TUNE_RED, TUNE_BLUE', colour)
        return
    title_str = ['red', 'blue', 'white'][colour]

    # Perform colour detection
    seg_mask = segment(
        image,
        [C_LOWER, B_LOWER][colour],
        [C_UPPER, B_UPPER][colour],
        invert=colour == TUNE_RED,
        title=f'{title_str} segment' if verbose else None
    )

    markers, stats, centroids = watershed(seg_mask)
    markers = analyse(
        markers, stats, centroids,
        ROCK_AREA_THRESHOLD
    )

    # Display results
    show(image, markers, centroids, title_str)
    return len(set(np.unique(markers)[1:]) - {1})


if __name__ == '__main__':
    # Initalisation
    logger.info('Initialising...')
    cap = cv2.VideoCapture(1) # 1 for external camera (USB)

    # Define socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # DGRAM for UDP

    # Define colours to detect
    blue = np.uint8([[[255, 0, 0]]])
    cyan = np.uint8([[[255, 255, 0]]])  # For detecting red via inverse HSV

    # Define lower and upper boundaries
    B_LOWER, B_UPPER = getBound(blue, var=30)
    C_LOWER, C_UPPER = getBound(cyan, var=10)

    logger.info('Ready.')

    prev_red, prev_blue = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        try:
            HEIGHT, WIDTH = frame.shape[:2]
        except AttributeError as error:
            logger.error('Unexpected frame %s throws AttributeError: %s', frame, error)
            continue
        if not ret:
            logger.warning('Unable to receive frame.')
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Detect red, blue, white
        red = find(frame, TUNE_RED, verbose=VERBOSE)
        blue = find(frame, TUNE_BLUE, verbose=VERBOSE)
        if red != prev_red or blue != prev_blue:
            logger.info(f'Red={red}; Blue={blue}')
            client_socket.sendto(bytes(f'Red={red}; Blue={blue}', 'utf-8'), (HOST, PORT))
            prev_red, prev_blue = red, blue

    logger.info('Exiting...')
    cap.release()
    cv2.destroyAllWindows()

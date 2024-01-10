import logging
import socket
from typing import Tuple

from tune import (
    H_MAX, S_MAX, V_MAX,
    TUNE_RED, TUNE_BLUE, TUNE_WHITE,
    segment
)

import cv2
from cv2.typing import MatLike
import numpy as np


# Define debugging
VERBOSE = False
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG if VERBOSE else logging.INFO)


# Define constants
PEBBLE_AREA_THRESHOLD = 150
ROCK_AREA_THRESHOLD = 500
HOST = '192.168.1.120'
PORT = 12345


# Define corners
TOP_LEFT_CORNER, TOP_RIGHT_CORNER, BOTTOM_LEFT_CORNER, BOTTOM_RIGHT_CORNER = range(4)
CAM_CORNER = TOP_RIGHT_CORNER


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
        #logger.info('Contour %i: Position (%i, %i), Area: %f', marker, x, y, area)

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
    # cv2.resizeWindow(title, 400, 400)

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

    # Label the disposal and airlock regions
    cv2.ellipse(disposal_mask, (cam_x, cam_y), DISPOSAL, 0, 0, 360, 0, 3)
    cv2.ellipse(disposal_mask, (cam_x, cam_y), AIRLOCK, 0, 0, 360, 0, 3)
    cv2.imshow(title, labelled_img)


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
    if not colour in (TUNE_RED, TUNE_BLUE, TUNE_WHITE):
        logger.error('colour=%i should be one of TUNE_RED, TUNE_BLUE, or TUNE_WHITE', colour)
        return
    title_str = ['red', 'blue', 'white'][colour]

    # Perform colour detection
    seg_mask = segment(
        image,
        [C_LOWER, B_LOWER, W_LOWER][colour],
        [C_UPPER, B_UPPER, W_UPPER][colour],
        invert=colour == TUNE_RED,
        hsl=colour == TUNE_WHITE,
        title=f'{title_str} segment' if verbose else None
    )

    markers, stats, centroids = watershed(seg_mask)
    markers = analyse(
        markers, stats, centroids,
        PEBBLE_AREA_THRESHOLD if colour == TUNE_WHITE else ROCK_AREA_THRESHOLD
    )

    """
    circles = cv2.HoughCircles(
        cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY),
        cv2.HOUGH_GRADIENT, 1, HEIGHT / 8,
        param1=30, param2=15,
        minRadius=20, maxRadius=30
    )
    if circles is None:
        return

    result = image.copy()
    circles = np.uint16(np.around(circles))
    for x, y, r, *_ in circles[0, :]:
        cv2.circle(result, (x, y), r, 255, 3)
    cv2.imshow(f'{title_str} result', result)
    """

    # Display results
    show(image, markers, centroids, title_str)
    return len(set(np.unique(markers)[1:]) - {1})


if __name__ == '__main__':
    # Initalisation
    logger.info('Initialising...')
    cap = cv2.VideoCapture(1)

    # Define socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    logger.info('Sending out connections on %s:%i', HOST, PORT)

    # Define colours to detect
    blue_colour = np.uint8([[[255, 0, 0]]])
    cyan = np.uint8([[[255, 255, 0]]])  # For detecting red via inverse HSV
    # white to be detected via HSL

    # Define lower and upper boundaries
    B_LOWER, B_UPPER = getBound(blue_colour, var=30)
    C_LOWER, C_UPPER = getBound(cyan, var=10)
    # W_LOWER, W_UPPER = getBound(white)
    W_LOWER = np.array([70, 180, 100])  # Manual boundary setting for white
    W_UPPER = np.array([255, 255, 255])

    logger.info('Ready.')

    while cap.isOpened():
        ret, frame = cap.read()
        try:
            HEIGHT, WIDTH = frame.shape[:2]
        except AttributeError as error:
            logger.error('Unexpected frame %s throws AttributeError: %s', frame, error)
            continue
        if not ret:
            logger.warning('Unable to receive frame.')
            continue
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cam_x = (WIDTH - 1) * (CAM_CORNER % 2)
        cam_y = (HEIGHT - 1) * int(CAM_CORNER >= BOTTOM_LEFT_CORNER)
        DISPOSAL = (610, 470)
        AIRLOCK = (1080, 875)

        # Create ellipses as boundary measurement
        disposal_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.ellipse(disposal_mask, (cam_x, cam_y), DISPOSAL, 0,  0, 360, 255, -1)
        disposal_img = cv2.bitwise_and(frame, frame, mask=disposal_mask)
        #cv2.imshow('disposal', disposal_img)

        airlock_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.ellipse(airlock_mask, (cam_x, cam_y), AIRLOCK, 0, 0, 360, 255, -1)
        #cv2.ellipse(airlock_mask, (cam_x, cam_y), DISPOSAL, 0, 0, 360, 0, -1)
        airlock_img = cv2.bitwise_and(frame, frame, mask=airlock_mask)
        #cv2.imshow('airlock', airlock_img)

        # Detect red, blue, white
        red = find(disposal_img, TUNE_RED, verbose=VERBOSE)
        blue = find(airlock_img, TUNE_BLUE, verbose=VERBOSE)
        white = find(disposal_img, TUNE_WHITE, verbose=VERBOSE)

        client_socket.send(f'RR={red};RB={blue};RW={white}'.encode())

    logger.info('Exiting...')
    cap.release()
    cv2.destroyAllWindows()

    """
    image = cv2.imread('assets/sample.png')
    HEIGHT, WIDTH = image.shape[:2]
    logger.info('Height: %i, Width: %i', HEIGHT, WIDTH)

    CAM_X = (WIDTH - 1) * (CAM_CORNER % 2)
    CAM_Y = (HEIGHT - 1) * int(CAM_CORNER >= BOTTOM_LEFT_CORNER)

    # Detect red, blue, white
    find(image, TUNE_RED, verbose=VERBOSE)
    find(image, TUNE_BLUE, verbose=VERBOSE)
    find(image, TUNE_WHITE, verbose=VERBOSE)

    logger.info('Exiting...')
    cv2.waitKey(0)
    """

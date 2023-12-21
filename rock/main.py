import logging
from typing import Tuple

from tune import H_MAX, S_MAX, V_MAX, segment

import cv2
from cv2.typing import MatLike
import numpy as np

logger = logging.getLogger(__name__)


# Define constants
PEBBLE_AREA_THRESHOLD = 250
ROCK_AREA_THRESHOLD = 300


# Define corners
TOP_LEFT_CORNER, TOP_RIGHT_CORNER, BOTTOM_LEFT_CORNER, BOTTOM_RIGHT_CORNER = range(4)
cam_corner = BOTTOM_RIGHT_CORNER


# Define debugging
VERBOSE = False
logging.basicConfig(level=logging.DEBUG if VERBOSE else logging.INFO)


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

    global height, width

    # Perform Gaussian blur on mask
    blur = cv2.GaussianBlur(mask, (7, 7), 2)

    # Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

    # Binarize gradient
    lowerb = np.array([0, 0, 0])
    upperb = np.array([15, 15, 15])
    binary = cv2.inRange(gradient, lowerb, upperb)

    # Flood fill from the edges
    for row in range(height):
        if binary[row, 0] == 255:
            cv2.floodFill(binary, None, (0, row), 0)
        if binary[row, width - 1] == 255:
            cv2.floodFill(binary, None, (width - 1, row), 0)
    for col in range(width):
        if binary[0, col] == 255:
            cv2.floodFill(binary, None, (col, 0), 0)
        if binary[height - 1, col] == 255:
            cv2.floodFill(binary, None, (col, height - 1), 0)

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
        area_threshold: int,
        region: bool
) -> MatLike:
    """Count number of detected rocks in image.

    The detected components are evaluated based on two criteria:
    - Area: The area of the component must meet the threshold.
    - Distance: The distance of the component from the corner of playfield must be smaller
                than the threshold(s) for airlock and/or disposal area.

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
    region : bool
        True if to detect in disposal, False to detect in airlock.

    Returns
    -------
    MatLike
        The markers of each pixel after filtering.
    """

    global height, width, cam_x, cam_y

    min_distance = (0 if region else width // 2) ** 2
    max_distance = (width // 2 if region else width) ** 2

    for marker, stat, centroid in zip(np.unique(markers)[1:], stats, centroids):
        # Obtain relevant statistics
        x, y = map(int, centroid)
        area = stat[cv2.CC_STAT_AREA]

        # Perform filtering
        distance = (x - cam_x) * (x - cam_x) + (y - cam_y) * (y - cam_y)
        if area < area_threshold or not min_distance <= distance <= max_distance:
            markers[markers == marker] = -1
        else:
            logger.info(
                'Contour %i: Position (%i, %i), Dist Squared %i, Area: %f',
                marker, x, y, distance, area
            )

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

    global height, width, cam_x, cam_y
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(title, 400, 400)

    # Assign the markers a hue between 0 and 179 (max H value)
    hue_markers = np.uint8(H_MAX * np.float32(markers) / np.max(markers))
    blank_channel = S_MAX * np.ones((height, width), dtype=np.uint8)
    marker_img = cv2.merge([hue_markers, blank_channel, blank_channel])
    marker_img = cv2.cvtColor(marker_img, cv2.COLOR_HSV2BGR)

    # Label the original image with the watershed markers
    labelled_img = image.copy()
    labelled_img[markers > 1] = marker_img[markers > 1]  # 1 is background color
    labelled_img = cv2.addWeighted(image, 0.5, labelled_img, 0.5, 0)

    # Label the centroid(s)
    logger.info('Centroids to label: %s', np.unique(markers))
    for marker in np.unique(markers)[1:]:
        if marker == 1:
            continue

        x, y = map(int, centroids[marker - 1])
        cv2.circle(
            labelled_img, (x, y),
            15, (255, 255, 255), 2
        )

    # Label the disposal and airlock regions
    cv2.circle(
        labelled_img, (cam_x, cam_y),
        height // 2, (0, 0, 0), 2
    )
    cv2.circle(
        labelled_img, (cam_x, cam_y),
        height, (0, 0, 0), 2
    )

    cv2.imshow(title, labelled_img)


def find_red_rock(
        image: MatLike,
        *,
        verbose: bool = False
) -> None:
    """Pipeline for finding red rock game elements.

    Parameters
    ----------
    image : MatLike
        The original image to find red rock game elements from.
    verbose : bool, default=False
        True to output the intermediate result(s), False otherwise.
    """

    global c_lower, c_upper

    seg_mask = segment(
        image, c_lower, c_upper,
        invert=True,
        title='red segmentation' if verbose else None
    )
    markers, stats, centroids = watershed(seg_mask)
    markers = analyse(
        markers, stats, centroids,
        ROCK_AREA_THRESHOLD, False
    )
    show(image, markers, centroids, 'red')


def find_blue_rock(
        image: MatLike,
        *,
        verbose: bool = False
) -> None:
    """Pipeline for finding blue rock game elements.

    Parameters
    ----------
    image : MatLike
        The original image to find blue rock game elements from.
    verbose : bool, default=False
        True to output the intermediate result(s), False otherwise.
    """

    global b_lower, b_upper

    seg_mask = segment(
        image, b_lower, b_upper,
        title='blue segmentation' if verbose else None
    )
    markers, stats, centroids = watershed(seg_mask)
    markers = analyse(
        markers, stats, centroids,
        ROCK_AREA_THRESHOLD, False
    )
    show(image, markers, centroids, 'blue')


def find_white_pebble(
        image: MatLike,
        *,
        verbose: bool = False
) -> None:
    """Pipeline for finding white pebble game elements.

    Parameters
    ----------
    image : MatLike
        The original image to find white pebble game elements from.
    verbose : bool, default=False
        True to output the intermediate result(s), False otherwise.
    """

    global w_lower, w_upper

    seg_mask = segment(
        image, w_lower, w_upper,
        hsl=True,
        title='white segmentation' if verbose else None
    )
    markers, stats, centroids = watershed(seg_mask)
    markers = analyse(
        markers, stats, centroids,
        PEBBLE_AREA_THRESHOLD, True
    )
    show(image, markers, centroids, 'white')


if __name__ == '__main__':
    # Initalisation
    logger.info('Initialising...')
    cap = cv2.VideoCapture(0)

    # Define colours to detect
    blue = np.uint8([[[255, 0, 0]]])
    cyan = np.uint8([[[255, 255, 0]]])  # For detecting red via inverse HSV
    # white to be detected via HSL

    # Define lower and upper boundaries
    b_lower, b_upper = getBound(blue)
    c_lower, c_upper = getBound(cyan, var=15)
    # w_lower, w_upper = getBound(white)
    w_lower = np.array([70, 180, 0])  # Manual boundary setting for white
    w_upper = np.array([179, 255, 255])

    logger.info('Ready.')

    while cap.isOpened():
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        if not ret:
            logger.warning('Unable to receive frame.')
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(height, width)
        cam_y = (height - 1) * int(cam_corner >= BOTTOM_LEFT_CORNER) # originally cam_x
        cam_x = (width - 1) * (cam_corner % 2) # originally cam_y

        # Detect red, blue, white
        find_red_rock(frame)
        find_blue_rock(frame)
        find_white_pebble(frame)

    logger.info('Exiting...')
    cap.release()
    cv2.destroyAllWindows()


    # image = cv2.imread('assets/sample.png')
    # height, width = image.shape[:2]
    # logger.info('Height: %i, Width: %i', height, width)

    # cam_x = (height - 1) * int(cam_corner >= BOTTOM_LEFT_CORNER)
    # cam_y = (width - 1) * (cam_corner % 2)

    # find_red_rock(image, verbose=VERBOSE)
    # find_blue_rock(image, verbose=VERBOSE)
    # find_white_pebble(image, verbose=VERBOSE)

    # logger.info('Exiting...')
    # cv2.waitKey(0)

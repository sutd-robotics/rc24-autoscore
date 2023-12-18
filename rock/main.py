import logging
from typing import Tuple

from tune import H_MAX, S_MAX, V_MAX

import cv2
from cv2.typing import MatLike
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Define constants
PEBBLE_AREA_THRESHOLD = 3000
ROCK_AREA_THRESHOLD = 5000
DISPOSAL_THRESHOLD = 0
AIRLOCK_THRESHOLD = float('inf')


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


def segment(
        image: MatLike,
        lower: np.ndarray,
        upper: np.ndarray,
        invert: bool = False
) -> MatLike:
    """Perform colour segmentation on a given image based on a lower and upper bound.

    Parameters
    ----------
    image : MatLike
        The image to perform segmentation on.
    lower : np.ndarray
        The lower bound of the colour to detect.
    upper : np.ndarray
        The upper bound of the colour to detect.
    invert : bool, default=False
        True if the HSV image is to be inverted, False otherwise.

    Returns
    -------
    MatLike
        The segmentation mask of the image.
    """

    # Convert the image to HSV, and invert if needed
    if invert:
        hsv_image = cv2.cvtColor(cv2.bitwise_not(image), cv2.COLOR_BGR2HSV)
    else:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Segment the image with the lower and upper bounds as mask
    mask = cv2.inRange(hsv_image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)


def watershed(
        mask: MatLike,
        height: int,
        width: int
) -> Tuple[int, MatLike, MatLike, MatLike]:
    """Perform watershed algorithm to detect connected contour components.

    Parameters
    ----------
    mask : MatLike
        The colour segmentation mask of the given image.
    height : int
        The height of the image, in pixels.
    width : int
        The width of the image, in pixels.

    Returns
    -------
    Tuple[int, MatLike, MatLike, MatLike]
        The number of labels present in the watershed mask,
        followed by the markers of each pixel,
        followed by the statistics of each of the markers,
        followed by the centroids of each of the markers.

    See Also
    --------
    https://stackoverflow.com/a/46084597
    https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
    """

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
    # Mark unknown regions as 0 and background regions as 1
    num_labels, markers, stats, centroids = cv2.connectedComponentsWithStats(foreground)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(mask, markers)

    return num_labels, markers, stats, centroids


def analyse(
        markers: MatLike,
        stats: MatLike,
        centroids: MatLike,
        area_threshold: int,
        min_distance: int = 0,
        max_distance: int = AIRLOCK_THRESHOLD
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
    min_distance : int, default=`0`
        The minimum distance from the corner of the playfield.
    max_distance : int, default=`AIRLOCK_THRESHOLD`
        The maximum distance from the corner of the playfield.

    Returns
    -------
    MatLike
        The markers of each pixel after filtering.
    """

    for i, (stat, centroid) in enumerate(zip(stats, centroids)):
        if i < 2:
            continue

        # Obtain relevant statistics
        x, y = map(int, centroid)
        area = stat[cv2.CC_STAT_AREA]

        # Perform filtering
        if not min_distance * min_distance <= x * x + y * y <= max_distance * max_distance or \
                area < area_threshold:
            #markers[markers == i] = 0
            pass
        logger.info('Contour %i: Position (%i, %i), Area: %f', i, x, y, area)

    return markers


def show(
        image: MatLike,
        markers: MatLike,
        centroids: MatLike,
        title: str,
        height: int,
        width: int
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
    height : int
        The height of the image.
    width : int
        The width of the image.
    """

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
    for i, centroid in enumerate(centroids):
        if i < 2 or markers[markers == i].size == 0:
            continue
        x, y = map(int, centroid)
        cv2.circle(labelled_img, (x, y), 15, (0, 0, 0), 2)

    cv2.imshow(title, labelled_img)


if __name__ == '__main__':
    # Initalisation
    logger.info('Initialising...')
    #cap = cv2.VideoCapture(0)

    # Define colours to detect
    blue = np.uint8([[[255, 0, 0]]])
    cyan = np.uint8([[[255, 255, 0]]])  # For detecting red via inverse HSV
    #white = np.uint8([[[255, 255, 255]]])

    # Define lower and upper boundaries
    b_lower, b_upper = getBound(blue)
    c_lower, c_upper = getBound(cyan)
    # w_lower, w_upper = getBound(white)
    w_lower = np.array([0, 0, 168])  # Manual boundary setting for white
    w_upper = np.array([172, 111, 255])

    # Define mapping
    mapping = {
        'red': ((c_lower, c_upper, True), ROCK_AREA_THRESHOLD, (DISPOSAL_THRESHOLD, AIRLOCK_THRESHOLD)),
        'blue': ((b_lower, b_upper, False), ROCK_AREA_THRESHOLD, (DISPOSAL_THRESHOLD, AIRLOCK_THRESHOLD)),
        'white': ((w_lower, w_upper, False), PEBBLE_AREA_THRESHOLD, (0, AIRLOCK_THRESHOLD))
    }
    logger.info('Ready.')

    """
    while cap.isOpened():
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        if not ret:
            logger.warning('Unable to receive frame.')
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Detect red, blue, white
        for colour, ((lower, upper, invert), area_thresh, (min_dist, max_dist)) in mapping.items():
            seg_mask = segment(frame, lower, upper, invert)
            _, markers, stats, centroids = watershed(seg_mask, height, width)
            markers = analyse(markers, stats, centroids, area_thresh, min_dist, max_dist)
            show(frame, markers, centroids, f'{colour} result', height, width)

    logger.info('Exiting...')
    cap.release()
    cv2.destroyAllWindows()
    """

    image = cv2.imread('assets/sample.png')
    height, width = image.shape[:2]
    for colour, ((lower, upper, invert), area_thresh, (min_dist, max_dist)) in mapping.items():
        seg_mask = segment(image, lower, upper, invert)
        _, markers, stats, centroids = watershed(seg_mask, height, width)
        markers = analyse(markers, stats, centroids, area_thresh, min_dist, max_dist)
        show(image, markers, centroids, f'{colour} result', height, width)

    logger.info('Exiting...')
    cv2.waitKey(0)

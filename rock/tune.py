import logging
from typing import Optional, Tuple

import cv2
from cv2.typing import MatLike
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Define constants
H_MAX = 179
S_MAX = V_MAX = L_MAX = 255


# Define labels
HMIN_LABEL = 'HMin'
SMIN_LABEL = 'SMin'
VMIN_LABEL = 'VMin'
LMIN_LABEL = 'LMin'
HMAX_LABEL = 'HMax'
SMAX_LABEL = 'SMax'
VMAX_LABEL = 'VMax'
LMAX_LABEL = 'LMax'
TITLE =  'image'


# Define debugging
TUNE_RED, TUNE_BLUE, TUNE_WHITE = range(3)
COLOUR = TUNE_WHITE


def init(colour: int = TUNE_BLUE) -> None:
    """Initialises the output window.

    Parameters
    ----------
    colour : int, default=`TUNE_BLUE`
        The colour to tune for.

    See Also
    --------
    https://stackoverflow.com/a/58194879
    """

    # Define constants
    nothing = lambda *_: None  # Create callable function
    min_label = LMIN_LABEL if colour == TUNE_WHITE else VMIN_LABEL
    max_label = LMAX_LABEL if colour == TUNE_WHITE else VMAX_LABEL
    value = L_MAX if colour == TUNE_WHITE else V_MAX

    # Create GUI overlay
    cv2.namedWindow(TITLE)
    cv2.createTrackbar(HMIN_LABEL, TITLE, 0, H_MAX, nothing)
    cv2.createTrackbar(SMIN_LABEL, TITLE, 0, S_MAX, nothing)
    cv2.createTrackbar(min_label, TITLE, 0, value, nothing)
    cv2.createTrackbar(HMAX_LABEL, TITLE, 0, H_MAX, nothing)
    cv2.createTrackbar(SMAX_LABEL, TITLE, 0, S_MAX, nothing)
    cv2.createTrackbar(max_label, TITLE, 0, value, nothing)

    # Set default value for max HSV / HSL trackbars
    cv2.setTrackbarPos(HMAX_LABEL, TITLE, H_MAX)
    cv2.setTrackbarPos(SMAX_LABEL, TITLE, S_MAX)
    cv2.setTrackbarPos(max_label, TITLE, value)


def update(colour: int = TUNE_BLUE) -> Tuple[np.ndarray, np.ndarray]:
    """Updates minimum and maximum HSV detection range based on trackbar positions.

    Parameters
    ----------
    colour : int, default=`TUNE_BLUE`
        The colour to tune for.

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

    # Define constants
    min_label = LMIN_LABEL if colour == TUNE_WHITE else VMIN_LABEL
    max_label = LMAX_LABEL if colour == TUNE_WHITE else VMAX_LABEL

    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos(HMIN_LABEL, TITLE)
    sMin = cv2.getTrackbarPos(SMIN_LABEL, TITLE)
    xMin = cv2.getTrackbarPos(min_label, TITLE)
    hMax = cv2.getTrackbarPos(HMAX_LABEL, TITLE)
    sMax = cv2.getTrackbarPos(SMAX_LABEL, TITLE)
    xMax = cv2.getTrackbarPos(max_label, TITLE)

    # Set minimum and maximum HSV / HLS values to display
    return (
        np.array([hMin, xMin, sMin]) if colour == TUNE_WHITE else np.array([hMin, sMin, xMin]),
        np.array([hMax, xMax, sMax]) if colour == TUNE_WHITE else np.array([hMax, sMax, xMax])
    )


def segment(
        image: MatLike,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        invert: bool = False,
        hsl: bool = False,
        title: Optional[str] = None
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
    hsl : bool, default=False
        True if to use HSL instead, False to use HSV.
    title : str, optional, default=None
        Title of window to show the result of the segmentation, if any.

    Returns
    -------
    MatLike
        The segmentation mask of the image.

    See Also
    --------
    https://stackoverflow.com/a/48184004
    """

    # Convert the image to HSV / HSL, and invert if needed
    res = cv2.cvtColor(
        cv2.bitwise_not(image) if invert else image,
        cv2.COLOR_BGR2HLS if hsl else cv2.COLOR_BGR2HSV
    )

    # Segment the image with the lower and upper bounds as mask
    mask = cv2.inRange(res, lower, upper)
    seg = cv2.bitwise_and(image, image, mask=mask)
    if title is not None:
        cv2.imshow(title, seg)  # For debugging

    return cv2.bitwise_and(image, image, mask=mask)


def process(image: MatLike, *, verbose: bool = False) -> None:
    """Processes a given image to segment and display selected ranges of colour.

    Parameters
    ----------
    image : MatLike
        The image to process.
    verbose : bool, default=False
        True if the segmentation mask should be printed, False otherwise.

    See Also
    --------
    `segment(image, lower, upper, *, invert, hsl, title)`
    """

    lower, upper = update(COLOUR)
    seg_mask = segment(
        image,
        lower, upper,
        invert=COLOUR == TUNE_RED,
        hsl=COLOUR == TUNE_WHITE,
        title=f'{["red", "blue", "white"][COLOUR]} segment' if verbose else None
    )
    cv2.imshow(TITLE, seg_mask)


if __name__ == '__main__':
    logger.info('Initialising...')
    init(COLOUR)
    verbose = False
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

        process(frame, verbose=verbose)

    logger.info('Exiting...')
    cap.release()
    """

    image = cv2.imread('assets/sample.png')
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        process(image, verbose=verbose)

    logger.info('Exiting...')
    cv2.destroyAllWindows()

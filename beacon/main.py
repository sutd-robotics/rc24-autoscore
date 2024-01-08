from concurrent.futures import Future, ThreadPoolExecutor
import logging
from typing import Mapping, Optional, Tuple, Union

import cv2
from cv2.typing import MatLike
import numpy as np
from qreader import QReader


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Define constants
RED = 'RC2024_red_flag'
BLUE = 'RC2024_blue_flag'


def process_qr(fn: Future) -> None:
    """Processes the result of decoding QR code(s) in a given image.

    The function expects and checks for (at most) two QR codes -
    one with content `RED` and one with content `BLUE`.

    Parameters
    ----------
    fn : Future
        The thread decoding the QR code(s) in the image.
        This function waits for the thread to complete and processes its result.

    See Also
    --------
    `decode_qr(image, detections)`
    """

    logger.info('Waiting for processed QR(s)...')
    res = fn.result()
    logger.info('Decoded %s', res)

    if RED in res:
        logger.info('Red detected')
    if BLUE in res:
        logger.info('Blue detected')


def decode_qr(
        image: MatLike,
        detections: Tuple[Mapping[str, Union[np.ndarray, float, Tuple[
            Union[float, int],
            Union[float, int]
        ]]], ...]
) -> Tuple[Optional[str], ...]:
    """Performs QR decoding on detected instance(s) of QR code(s) in a given image.

    This function should be run asynchronously as the decoding job is computationally intensive.

    Parameters
    ----------
    image : MatLike
        The given image to process and decode scanned QR code(s) on.
    detections : Tuple[Mapping[str, Union[np.ndarray, float, Tuple[
                     Union[float, int],
                     Union[float, int]
                 ]]], ...]
        The detected QR code(s) and their respective statistics.

    Returns
    -------
    Tuple[Optional[str], ...]
        The content(s) of the decoded QR code(s), if any.

    See Also
    --------
    https://pypi.org/project/qreader/
    """

    logger.info('Decoding QR(s)...')
    return tuple(QREADER.decode(image, detection) for detection in detections)


def detect_qr(image: MatLike) -> None:
    """Performs QR detection in a given image.

    If QR code(s) is/are detected, the function spawns another thread
    and performs the decoding / processing job asynchronously.

    Parameters
    ----------
    image : MatLike
        The given image to detect QR code(s) on.

    See Also
    --------
    `decode_qr(image, detections)`
    """

    # Perform preliminary detection
    detections = QREADER.detect(image, is_bgr=True)
    if detections:

        # If QR code(s) is/are detected, perform (heavy) decoding
        decode_fn = EXECUTOR.submit(decode_qr, image, detections)
        _ = EXECUTOR.submit(process_qr, decode_fn)

        # Obtain and draw bounding box
        for detection_stats in detections:
            x1, y1, x2, y2 = map(int, detection_stats['bbox_xyxy'])
            cv2.rectangle(
                image,
                (x1, y1), (x2, y2),
                (0, 0, 0),
                thickness=5
            )

    cv2.imshow('result', image)


if __name__ == '__main__':
    # Initialisation
    logger.info('Initialising...')
    QREADER = QReader(min_confidence=0.7)
    EXECUTOR = ThreadPoolExecutor()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # this is the magic!

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(cap.read()[1].shape)
    #red = blue = False
    logger.info('Ready.')

    cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning('Unable to receive frame.')
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detect_qr(frame)

    logger.info('Exiting...')
    EXECUTOR.shutdown(wait=False, cancel_futures=True)
    cap.release()
    cv2.destroyAllWindows()

    """
    detect_qr(cv2.imread('assets/sample.png'))

    logger.info('Exiting...')
    cv2.waitKey(0)
    """

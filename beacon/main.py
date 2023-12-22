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

    logger.info('Waiting for processed QR(s)...')
    res = fn.result()
    logger.info('Decoded %s', res)

    if RED in res:
        logger.info('Red detected')
    if BLUE in res:
        logger.info('Blue detected')


def decode_qr(
        image: MatLike,
        detections: Tuple[Mapping[str, Union[np.ndarray, float, Tuple[Union[float, int], Union[float, int]]]], ...]
) -> Tuple[Optional[str], ...]:

    global QREADER

    logger.info('Decoding QR(s)...')
    return tuple(QREADER.decode(image, detection) for detection in detections)


if __name__ == '__main__':
    # Initialisation
    logger.info('Initialising...')
    QREADER = QReader(min_confidence=0.7)
    EXECUTOR = ThreadPoolExecutor()
    cap = cv2.VideoCapture(0)
    #red = blue = False
    logger.info('Ready.')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning('Unable to receive frame.')
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Perform preliminary detection
        detections = QREADER.detect(frame, is_bgr=True)
        if detections:

            # If QR code(s) is/are detected, perform (heavy) decoding
            decode_fn = EXECUTOR.submit(decode_qr, frame, detections)
            _ = EXECUTOR.submit(process_qr, decode_fn)

            # Obtain and draw bounding box
            for detection_stats in detections:
                x1, y1, x2, y2 = map(int, detection_stats['bbox_xyxy'])
                cv2.rectangle(
                    frame,
                    (x1, y1), (x2, y2),
                    (0, 0, 0),
                    thickness=5
                )

        cv2.imshow('result', frame)

    logger.info('Exiting...')
    EXECUTOR.shutdown(wait=False, cancel_futures=True)
    cap.release()
    cv2.destroyAllWindows()

    """
    process_qr(cv2.imread('assets/sample.png'), QREADER)

    logger.info('Exiting...')
    cv2.waitKey(0)
    """

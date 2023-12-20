import logging

import cv2
from cv2.typing import MatLike
from qreader import QReader


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Define constants
RED = 'Red'
BLUE = 'Blue'


def process_qr(image: MatLike, qreader: QReader) -> None:
    """Performs QR detection and decoding on a given image.

    Two QR codes are expected - one with content `RED` and one with content `BLUE`.
    The function draws an appropriately-coloured bounding box around the QR code when detected.

    Parameters
    ----------
    image : MatLike
        The image to perform QR detection on.
    qreader : QReader
        `QReader` ML class for QR detection and decoding.

    See Also
    --------
    https://pypi.org/project/qreader/
    """

    # Perform QR decoding
    detections = qreader.detect_and_decode(
        image,
        return_detections=True,
        is_bgr=True
    )

    # Process results
    for i, (decoded_content, detection_stats) in enumerate(zip(*detections)):
        logger.info(
            'QR Code %i - Confidence: %f, Content: %s',
            i + 1, detection_stats['confidence'], decoded_content
        )

        # Obtain and draw bounding box
        x1, y1, x2, y2 = map(int, detection_stats['bbox_xyxy'])
        cv2.rectangle(
            image,
            (x1, y1), (x2, y2),
            (255 * int(decoded_content == BLUE), 0, 255 * int(decoded_content == RED)),
            thickness=5
        )

    cv2.imshow('result', image)


if __name__ == '__main__':
    # Initialisation
    logger.info('Initialising...')
    QREADER = QReader(min_confidence=0.7)
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

        process_qr(frame, QREADER)

    logger.info('Exiting...')
    cap.release()
    cv2.destroyAllWindows()
    """

    process_qr(cv2.imread('assets/sample.png'), QREADER)

    logger.info('Exiting...')
    cv2.waitKey(0)

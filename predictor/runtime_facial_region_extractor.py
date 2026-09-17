from pathlib import Path

import numpy as np
import cv2 as cv

def adjust_coordinates(input_size, y_start, y_end, x_start, x_end) -> tuple[int, int, int, int]:
    """Adjusts coordinates so that no points are out of bounds.

    Args:
        input_size (int): Height and width of input image.
        y_start (int): Lower y point.
        y_end (int): Higher y point.
        x_start (int): Lower x point.
        x_end (int): Higher x point.

    Returns:
        tuple[int, int, int, int]: Tuple containing the corrected coordinates.
    """

    new_y_start, new_y_end = y_start, y_end
    new_x_start, new_x_end = x_start, x_end
    if new_y_start < 0:
        shift = abs(new_y_start)
        new_y_start += shift
        new_y_end += shift
    if new_y_end > input_size:
        shift = new_y_end - input_size
        new_y_start -= shift
        new_y_end -= shift
    if new_x_start < 0:
        shift = abs(new_x_start)
        new_x_start += shift
        new_x_end += shift
    if new_x_end > input_size:
        shift = new_x_end - input_size
        new_x_start -= shift
        new_x_end -= shift
    return new_y_start, new_y_end, new_x_start, new_x_end

class FacialRegionExtractor:
    """Class for handling facial region extraction from images.

    Attributes:
        input_list (list[path]): List of paths to input images.
        index (int): Current index within input_list.
        input_size (int): Height and width of input images.
        output_size (int): Height and width of output images.
        yunet (FaceDetectorYN): YuNet detector for image recognition.
    """

    def __init__(self, input, input_size = 320, output_size = 32):
        input_path = Path(f'./{input}/')
        fake_list = list(input_path.glob('fake/stylegan/*/*'))
        real_list = list(input_path.glob('real/population_split/*/*/*'))
        self.input_list = fake_list + real_list
        self.index = 0
        self.input_size = input_size
        self.output_size = output_size
        self.yunet = cv.FaceDetectorYN.create(
            'face_detection_yunet_2026may.onnx',  # model
            "",                                   # config
            (input_size, input_size),             # input_size
            0.85,                                 # score_threshold
            0.3,                                  # nms_threshold
            5000                                  # top_k
        )

    def next_processed_image(self, feature, increment=1) -> np.ndarray | None:
        """Returns the next image in self.input_list, cropped to feature.

        Args:
            feature (int): The feature to be extracted.
                0 = Right Eye
                1 = Left Eye
                2 = Nose
                3 = Mouth
            increment (int): Increments self.index if 1, does not if 0.

        Returns:
            np.ndarray: NumPy array representing the cropped image.
            None: Nothing, if the image could not be read.
        """

        image_path = self.input_list[self.index]
        if increment: self.index += 1
        print(f'Opening image at {image_path}')

        # Grab image and resize.
        img = cv.imread(image_path)
        if type(img) is not np.ndarray:
            print(f'\t{img} not read as numpy.ndarray, returning.')
            return

        img = cv.resize(img, (self.input_size, self.input_size))
        width, height, _ = img.shape
        self.yunet.setInputSize((height, width))

        # Run image through YuNet for face detection.
        face = self.yunet.detect(img)
        if face[1] is None: return
        x1 = int(face[1][0][0])         # top-left coordinate, x
        y1 = int(face[1][0][1])         # top-left coordinate, y
        w = int(face[1][0][2])          # width of face
        h = int(face[1][0][3])          # height of face
        x_re = int(face[1][0][4])       # right eye, x
        y_re = int(face[1][0][5])       # right eye, y
        x_le = int(face[1][0][6])       # left eye, x
        y_le = int(face[1][0][7])       # left eye, y
        x_nt = int(face[1][0][8])       # nose tip, x
        y_nt = int(face[1][0][9])       # nose tip, y
        x_rcm = int(face[1][0][10])     # right mouth corner, x
        y_rcm = int(face[1][0][11])     # right mouth corner, y
        x_lcm = int(face[1][0][12])     # left mouth corner, x
        y_lcm = int(face[1][0][13])     # left mouth corner, y

        # Snip image to only requested feature, then return the image.
        if feature == 0:
            y_start, y_end = y_re - self.output_size, y_re + self.output_size
            x_start, x_end = x_re - self.output_size, x_re + self.output_size
            y_start, y_end, x_start, x_end = adjust_coordinates(self.input_size, 
                                                                y_start, y_end, x_start, x_end)
            right_eye_img = img[y_start:y_end, x_start:x_end]
            print(f'\tReturning right eye image')
            return right_eye_img

        if feature == 1:
            y_start, y_end = y_le - self.output_size, y_le + self.output_size
            x_start, x_end = x_le - self.output_size, x_le + self.output_size
            y_start, y_end, x_start, x_end = adjust_coordinates(self.input_size, 
                                                                y_start, y_end, x_start, x_end)
            left_eye_img = img[y_start:y_end, x_start:x_end]
            print(f'\tReturning left eye image')
            return left_eye_img

        if feature == 2:
            y_start, y_end = y_nt - self.output_size, y_nt + self.output_size
            x_start, x_end = x_nt - self.output_size, x_nt + self.output_size
            y_start, y_end, x_start, x_end = adjust_coordinates(self.input_size, 
                                                                y_start, y_end, x_start, x_end)
            nose_img = img[y_start:y_end, x_start:x_end]
            print(f'\tReturning nose image')
            return nose_img
            
        if feature == 3:
            y_center = int((y_rcm + y_lcm) / 2)
            x_center = int((x_rcm + x_lcm) / 2)
            y_start, y_end = y_center - self.output_size, y_center + self.output_size
            x_start, x_end = x_center - self.output_size, x_center + self.output_size
            y_start, y_end, x_start, x_end = adjust_coordinates(self.input_size, 
                                                                y_start, y_end, x_start, x_end)
            mouth_img = img[y_start:y_end, x_start:x_end]
            print(f'\tReturning mouth image')
            return mouth_img

if __name__ == "__main__":
    fre = FacialRegionExtractor(input='input')
    for i in range(0,5):
        cv.imshow("img", fre.next_processed_image(feature=0))
        cv.waitKey(0)

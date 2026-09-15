import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='input', help='path to input folder.')
parser.add_argument('-o', '--output', type=str, default='output', help='path to output folder. will be filled with identical folder structure as input.')
parser.add_argument('-f', '--feature', type=str, help='feature to extract. must be \'all\', \'eyes\', \'nose\', or \'mouth\'')
parser.add_argument('-iS', '--input_size', type=int, default=320, help='height and width of provided images.')
parser.add_argument('-oS', '--output_size', type=int, default=32, help='distance from center in pixels for extracted images. i.e. \'32\' would return 64x64 features.')
args = parser.parse_args()

if args.feature != "all" and args.feature != "eyes" and args.feature != "nose" and args.feature != "mouth":
    print('Must provide feature: \'all\', \'eyes\', \'nose\', or \'mouth\'')
    quit()

check_eyes = True if args.feature == "eyes" or args.feature == "all" else False
check_nose = True if args.feature == "nose" or args.feature == "all" else False
check_mouth = True if args.feature == "mouth" or args.feature == "all" else False

# Create list of paths to all images
# NOTE - This assumes that the input directory structure is identical to that of
#   the 'final' dataset created by Craig and Kaden.
input_path = Path(f'./{args.input}/')
fake_list = list(input_path.glob('fake/stylegan/*/*'))
real_list = list(input_path.glob('real/population_split/*/*/*'))
input_list = fake_list + real_list

# Create mirrored directories in output.
# NOTE - This assumes that the input directory structure is identical to that of
#   the 'final' dataset created by Craig and Kaden.
fake_dir_list = list(input_path.glob('fake/stylegan/*'))
real_dir_list = list(input_path.glob('real/population_split/*/*'))
input_dir_list = fake_dir_list + real_dir_list
for dir in input_dir_list:
    for feature in ['right_eye', 'left_eye', 'nose', 'mouth']:
        output_dir_string = f'./{args.output}/{feature}{str(dir)[5:]}'
        output_dir = Path(output_dir_string)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

# Initialize YuNet detector.
detector = cv.FaceDetectorYN.create(
    'face_detection_yunet_2026may.onnx',  # model
    "",                                   # config
    (args.input_size, args.input_size),   # input_size
    0.85,                                 # score_threshold
    0.3,                                  # nms_threshold
    5000                                  # top_k
)

# For each image provided as input...
for img_path in input_list:
    print(f'Opening image at {img_path}')

    # Grab image and resize.
    img = cv.imread(img_path)
    if type(img) is not np.ndarray:
        print(f'\t{img} not read as numpy.ndarray, skipping.')
        continue

    img = cv.resize(img, (args.input_size, args.input_size))
    width, height, _ = img.shape
    detector.setInputSize((height, width))

    # Run image through YuNet for face detection.
    face = detector.detect(img)
    if face[1] is None: continue
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

    # Snip image to only requested feature(s), then save to disk.
    if check_eyes:
        y_start, y_end = y_re - args.output_size, y_re + args.output_size
        x_start, x_end = x_re - args.output_size, x_re + args.output_size
        y_start, y_end, x_start, x_end = adjust_coordinates(args.input_size, 
                                                            y_start, y_end, x_start, x_end)
        right_eye_img = img[y_start:y_end, x_start:x_end]
        final_path = f'{args.output}/right_eye{str(img_path)[5:]}'
        print(f'\tPrinting right eye to {final_path}')
        cv.imwrite(final_path, right_eye_img)

        y_start, y_end = y_le - args.output_size, y_le + args.output_size
        x_start, x_end = x_le - args.output_size, x_le + args.output_size
        y_start, y_end, x_start, x_end = adjust_coordinates(args.input_size, 
                                                            y_start, y_end, x_start, x_end)
        left_eye_img = img[y_start:y_end, x_start:x_end]
        final_path = f'{args.output}/left_eye{str(img_path)[5:]}'
        print(f'\tPrinting left eye to {final_path}')
        cv.imwrite(final_path, left_eye_img)

    if check_nose:
        y_start, y_end = y_nt - args.output_size, y_nt + args.output_size
        x_start, x_end = x_nt - args.output_size, x_nt + args.output_size
        y_start, y_end, x_start, x_end = adjust_coordinates(args.input_size, 
                                                            y_start, y_end, x_start, x_end)
        nose_img = img[y_start:y_end, x_start:x_end]
        final_path = f'{args.output}/nose{str(img_path)[5:]}'
        print(f'\tPrinting nose to {final_path}')
        cv.imwrite(final_path, nose_img)
        
    if check_mouth:
        y_center = int((y_rcm + y_lcm) / 2)
        x_center = int((x_rcm + x_lcm) / 2)
        y_start, y_end = y_center - args.output_size, y_center + args.output_size
        x_start, x_end = x_center - args.output_size, x_center + args.output_size
        y_start, y_end, x_start, x_end = adjust_coordinates(args.input_size, 
                                                            y_start, y_end, x_start, x_end)
        mouth_img = img[y_start:y_end, x_start:x_end]
        final_path = f'{args.output}/mouth{str(img_path)[5:]}'
        print(f'\tPrinting mouth to {final_path}')
        cv.imwrite(final_path, mouth_img)

print("*******************************")
print("* FEATURE EXTRACTION COMPLETE *")
print("*******************************")

import glob
from PIL import Image, ImageDraw, ImageEnhance
import os
import random
import numpy as np

flip_probability = 0.5
rotation_probability = 0.25
crop_probability = 0.25
min_contrast = 0.5
max_contrast = 1.5
min_brightness = 0.5
max_brightness = 1.5
min_saturation = 0.5
max_saturation = 1.5
min_rotation = -30.0
max_rotation = 30.0

def visualize_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(xy=((box[0], box[1]), (box[2], box[3])), outline='red')

def load_annotation_data(annotation_path):
    boxes = []
    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            line = line.split()
            boxes.append((int(line[0]), int(line[1]), int(line[2]), int(line[3])))
    return boxes

def save_annotation_data(annotation_path, boxes):
    with open(annotation_path, 'w') as f:
        for box in boxes:
            f.write('{} {} {} {} 0 -1 nomask 0 0\n'.format(box[0], box[1], box[2], box[3]))

def augment_flip(image, boxes):
    if random.random() >= flip_probability:
        return image, boxes
    augmented_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_boxes = []
    for box in boxes:
        augmented_box = (image.width - box[0] - 1, box[1], image.width - box[2] - 1, box[3])
        augmented_boxes.append(augmented_box)
    return augmented_image, augmented_boxes

def rotate_around_point(point, origin, angle):
    px = point[0] - origin[0]
    py = point[1] - origin[1]
    s = np.sin(-angle)
    c = np.cos(-angle)
    xnew = px * c - py * s + origin[0]
    ynew = px * s + py * c + origin[1]
    return (xnew, ynew)

def make_aabb(points):
    xmin = min(points[0][0], points[1][0], points[2][0], points[3][0])
    ymin = min(points[0][1], points[1][1], points[2][1], points[3][1])
    xmax = max(points[0][0], points[1][0], points[2][0], points[3][0])
    ymax = max(points[0][1], points[1][1], points[2][1], points[3][1])
    return (xmin, ymin, xmax, ymax)

def clamp(x, min, max):
    if x < min:
        x = min
    if x > max:
        x = max
    return x

def clamp_aabb(box, width, height):
    return (clamp(box[0], 0, width - 1), clamp(box[1], 0, height - 1), clamp(box[2], 0, width - 1), clamp(box[3], 0, height - 1))

def augment_rotation(image, boxes):
    if random.random() >= rotation_probability:
        return image, boxes
    deg_angle = random.uniform(min_rotation, max_rotation)
    angle = np.deg2rad(deg_angle)
    image = image.rotate(deg_angle)
    # Rotate the boxes
    center = (0.5 * image.width, 0.5 * image.height)
    out_boxes = []
    for box in boxes:
        top_left     = rotate_around_point((box[0], box[1]), center, angle)
        top_right    = rotate_around_point((box[2], box[1]), center, angle)
        bottom_left  = rotate_around_point((box[0], box[3]), center, angle)
        bottom_right = rotate_around_point((box[2], box[3]), center, angle)
        out_boxes.append(clamp_aabb(make_aabb((top_left, top_right, bottom_left, bottom_right)), image.width, image.height))
    return image, out_boxes

def augment_crop(image, boxes):
    if random.random() >= crop_probability:
        return image, boxes
    xmin = random.randint(0, image.width // 4)
    xmax = random.randint(3 * image.width // 4, image.width - 1)
    ymin = random.randint(0, image.height // 4)
    ymax = random.randint(3 * image.height // 4, image.height - 1)
    image = image.crop((xmin, ymin, xmax, ymax))
    # Adjust the boxes
    out_boxes = []
    for box in boxes:
        out_boxes.append(clamp_aabb((box[0] - xmin, box[1] - ymin, box[2] - xmin, box[3] - ymin), image.width, image.height))
    return image, out_boxes

def augment_brightness(image, boxes):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(min_brightness, max_brightness)), boxes

def augment_saturation(image, boxes):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(random.uniform(min_saturation, max_saturation)), boxes

def augment_contrast(image, boxes):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(random.uniform(min_contrast, max_contrast)), boxes

def create_augmentation_for_single_item(image_path, path_to_output, num_augmentations):
    image_filename = os.path.basename(image_path)
    annotation_path = image_path.rsplit(".", 1)[0] + '.gt_data.txt'
    annotation_filename = os.path.basename(annotation_path)
    boxes = load_annotation_data(annotation_path)
    image = Image.open(image_path)
    for i in range(num_augmentations):
        augmented_image, augmented_boxes = augment_flip(image, boxes)
        augmented_image, augmented_boxes = augment_rotation(augmented_image, augmented_boxes)
        augmented_image, augmented_boxes = augment_crop(augmented_image, augmented_boxes)
        augmented_image, augmented_boxes = augment_saturation(augmented_image, augmented_boxes)
        augmented_image, augmented_boxes = augment_contrast(augmented_image, augmented_boxes)
        augmented_image, augmented_boxes = augment_brightness(augmented_image, augmented_boxes)
        # Uncomment if you want to visualize the boxes
        #visualize_boxes(augmented_image, augmented_boxes)
        augmented_image.save(path_to_output + '/' + image_filename.rsplit(".", 1)[0] + '_aug' + f'{i:03d}' + '.jpg')
        save_annotation_data(path_to_output + '/' + image_filename.rsplit(".", 1)[0] + '_aug' + f'{i:03d}' + '.gt_data.txt', augmented_boxes)

def augmentation(path_to_dataset, path_to_output, num_augmentations):
    image_filenames = [f for f in glob.glob(path_to_dataset + "/*.jpg")]
    for filename in image_filenames:
        create_augmentation_for_single_item(filename, path_to_output, num_augmentations)

augmentation('dataset_small', 'dataset_augmented', 5)

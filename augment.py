import glob
from PIL import Image, ImageDraw
import os

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

def augment_transpose(image, boxes):
    augmented_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_boxes = []
    for box in boxes:
        augmented_box = box
        augmented_boxes.append(augmented_box)
    return augmented_image, augmented_boxes

def create_augmentation_for_single_item(image_path, path_to_output, num_augmentations):
    image_filename = os.path.basename(image_path)
    annotation_path = image_path.rsplit(".", 1)[0] + '.gt_data.txt'
    annotation_filename = os.path.basename(annotation_path)
    boxes = load_annotation_data(annotation_path)
    image = Image.open(image_path)
    for i in range(num_augmentations):
        augmented_image, augmented_boxes = augment_transpose(image, boxes)
        visualize_boxes(augmented_image, augmented_boxes)
        augmented_image.save(path_to_output + '/' + image_filename.rsplit(".", 1)[0] + '_aug' + f'{i:03d}' + '.jpg')

def augmentation(path_to_dataset, path_to_output, num_augmentations):
    image_filenames = [f for f in glob.glob(path_to_dataset + "/*.jpg")]
    for filename in image_filenames:
        create_augmentation_for_single_item(filename, path_to_output, num_augmentations)

augmentation('dataset_small', 'dataset_augmented', 1)

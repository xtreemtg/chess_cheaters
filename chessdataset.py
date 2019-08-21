from mrcnn import utils
from skimage.io import imread
import numpy as np
import cv2
import json

CLASS_DICT = {'white_bishop': 1, 'black_bishop': 2, 'white_king': 3, 'black_king': 4,
              'white_queen': 5, 'black_queen': 6, 'white_pawn': 7, 'black_pawn': 8,
              'white_castle': 9, 'black_castle': 10, 'white_knight': 11, 'black_knight': 12}


class ChessDataset(utils.Dataset):

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pieces', 1, 'white_bishop')
        self.add_class('pieces', 2, 'black_bishop')
        self.add_class('pieces', 3, 'white_king')
        self.add_class('pieces', 4, 'black_king')
        self.add_class('pieces', 5, 'white_queen')
        self.add_class('pieces', 6, 'black_queen')
        self.add_class('pieces', 7, 'white_pawn')
        self.add_class('pieces', 8, 'black_pawn')
        self.add_class('pieces', 9, 'white_castle')
        self.add_class('pieces', 10, 'black_castle')
        self.add_class('pieces', 11, 'white_knight')
        self.add_class('pieces', 12, 'black_knight')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pieces', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = imread(fp)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info.get('annotations', [])
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                polyline_dict = json.loads(annotations[i].get('region_shape_attributes', []))
                x_points = 0.1*np.array(polyline_dict['all_points_x'])
                y_points = 0.1*np.array(polyline_dict['all_points_y'])
                vertices = np.array([x for x in zip(x_points, y_points)], 'int32')
                cv2.fillConvexPoly(mask[:, :, i], vertices, 255)
                class_ids[i] = CLASS_DICT[json.loads(annotations[i]['region_attributes'])['name']]
        return mask.astype(np.bool), class_ids.astype(np.int32)

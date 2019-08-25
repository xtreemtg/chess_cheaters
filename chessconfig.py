from mrcnn.config import Config


class ChessConfig(Config):
    # Give the configuration a recognizable name
    NAME = 'chess_config'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # BACKBONE = 'resnet50'

    NUM_CLASSES = 13  # background + 12 pieces classes

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 448
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01

    STEPS_PER_EPOCH = 200



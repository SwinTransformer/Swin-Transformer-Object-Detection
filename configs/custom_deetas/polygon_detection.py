# the new config inherits the base configs to highlight the necessary modification
# _base_ = '../swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
# _base_ = '../swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
_base_ = '../swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
# _base_ = '../swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'

HOME_dir = '/maeng_space'
annotation_root_dir = HOME_dir + '/Data_and_output/Deetas/22_04_12/json_MaskRCNN_large_02/'
image_root_dir = HOME_dir + '/Data_and_output/Deetas/image_all/'

# IMAGE_SCALE = (100, 100)

NUM_CLASSES = 9

# data_type = 'bounding_box'
# CLASSE_NAMES = ('Person', 'Vehicle', 'WheeledObject', 'MovableObject')

data_type = 'segmentation'
CLASSE_NAMES = ('FixedObject', 'Obstruction', 'AutomaticDoor', 'HingerDoor', 'Elevator',
            'Address', 'Sign', 'Screen', 'Handle')

# data_type = 'static_action'
# CLASSE_NAMES = ('AutomaticdoorOpening', 'AutomaticdoorClosing', 'HingerdoorOpening', 'HingerdoorClosing', 'ElevatorOpening',
#             'ElevatorClosing')


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,

    train=dict(
        type=dataset_type,
        classes=CLASSE_NAMES,
        ann_file= annotation_root_dir + data_type + '_train.json',
        img_prefix= image_root_dir),
    val=dict(
        type=dataset_type,
        classes=CLASSE_NAMES,
        ann_file= annotation_root_dir + data_type + '_val.json',
        img_prefix=image_root_dir),
    test=dict(
        type=dataset_type,
        classes=CLASSE_NAMES,
        ann_file= annotation_root_dir + data_type + '_test.json',
        img_prefix=image_root_dir))


# train_pipeline = [
#     dict(type='Resize', img_scale=IMAGE_SCALE, keep_ratio=True),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=IMAGE_SCALE,
#         )
# ]

# 2. model settings

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=NUM_CLASSES,
                ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=NUM_CLASSES,
                ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=NUM_CLASSES,
                )
        ],
        mask_head=dict(
            num_classes=NUM_CLASSES,
            )),)

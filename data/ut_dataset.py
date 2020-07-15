import os
import xml.etree.ElementTree as ET
import json
import numpy as np

from .util import read_image


class UTDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval_new',
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        id_list_file = os.path.join(
            data_dir, 'ut_set/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = UT_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]

        anno =os.path.join(self.data_dir, 'ut_tidy_anno/add_interact_x', id_ + '.json')
        interact_bbox = list()
        interact_label = list()
        interact_difficult = list()
        with open(anno, 'r') as f:
            frame_info = json.load(f)
        # 打开这个文件，并取出coor信息
        for coor_info in frame_info['coorlist']:

            coordinate = coor_info['coor']

            coor_num=len(coordinate)
            # 转str为int，并减掉1
            coordinate = list(map(float, coordinate))
            coordinate = list(map(lambda x: x - 1, coordinate))
            coor_ = []
            # 换位
            coor_.append(coordinate[1])
            coor_.append(coordinate[0])
            coor_.append(coordinate[3])
            coor_.append(coordinate[2])
            if coor_num==5:
                coor_.append(coordinate[4])
            # 获取该box的action
            action = coor_info['action']

            if action == 'nad':
                action = 'na'
            # 将单人框和交互框做分开处理
            if action =='interact':
                interact_bbox.append(coor_)
                interact_label.append(11)
                interact_difficult.append(0)

        #如果没有interact动作，转化为numpy array的方法将改变
        interact_bbox = np.stack(interact_bbox).astype(np.float32)
        interact_label = np.stack(interact_label).astype(np.int32)
        interact_difficult = np.array(interact_difficult, dtype=np.bool).astype(np.uint8)

        # Load a image
        img_file = os.path.join(self.data_dir, 'frame', id_ + '.jpg')
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, interact_bbox, interact_label,interact_difficult
    __getitem__ = get_example

UT_BBOX_LABEL_NAMES = (
    'interact'
)

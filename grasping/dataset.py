from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
import random


# rotate_idx can be 0 = 90, 1 = 180, 2 ..., 3...
# rotates CCW
def rotate_around_origin_90s(pos: np.array, rotate_idx: int) -> np.array:
    assert pos.shape == (2,)
    if rotate_idx == 0:
        return np.array([pos[0], pos[1]])
    elif rotate_idx == 1:
        return np.array([pos[1], -pos[0]])
    elif rotate_idx == 2:
        return np.array([-pos[0], -pos[1]])
    elif rotate_idx == 3:
        return np.array([-pos[1], pos[0]])
    else:
        assert False

class GraspDataset(Dataset):
    def __init__(self, train: bool=True) -> None:
        '''Dataset of successful grasps.  Each data point includes a 64x64
        top-down RGB image of the scene and a grasp pose specified by the gripper
        position in pixel space and rotation (either 0 deg or 90 deg)

        The datasets are already created for you, although you can checkout
        `collect_dataset.py` to see how it was made (this can take a while if you
        dont have a powerful CPU).
        '''
        mode = 'train' if train else 'val'
        self.train = train
        data = np.load(f'{mode}_dataset.npz')
        self.imgs = data['imgs']
        self.actions = data['actions']

    def transform_grasp(self, img: Tensor, action: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        '''Randomly rotate grasp by 0, 90, 180, or 270 degrees.  The image can be
        rotated using `TF.rotate`, but you will have to do some math to figure out
        how the pixel location and gripper rotation should be changed.

        Arguments
        ---------
        img:
            float tensor ranging from 0 to 1, shape=(3, 64, 64)
        action:
            array containing (px, py, rot_id), where px specifies the row in
            the image (heigh dimension), py specifies the column in the image (width dimension),
            and rot_id is an integer: 0 means 0deg gripper rotation, 1 means 90deg rotation.

        Returns
        -------
        tuple of (img, action) where both have been transformed by random
        rotation in the set (0 deg, 90 deg, 180 deg, 270 deg)

        Note
        ----
        The gripper is symmetric about 180 degree rotations so a 180deg rotation of
        the gripper is equivalent to a 0deg rotation and 270 deg is equivalent to 90 deg.

        Example Action Rotations
        ------------------------
        action = (32, 32, 1)
         - Rot   0 deg : rot_action = (32, 32, 1)
         - Rot  90 deg : rot_action = (32, 32, 0)
         - Rot 180 deg : rot_action = (32, 32, 1)
         - Rot 270 deg : rot_action = (32, 32, 0)

        action = (0, 63, 0)
         - Rot   0 deg : rot_action = ( 0, 63, 0)
         - Rot  90 deg : rot_action = ( 0,  0, 1)
         - Rot 180 deg : rot_action = (63,  0, 0)
         - Rot 270 deg : rot_action = (63, 63, 1)
        '''
        assert img.shape == (3,64,64)

        random_rot_selection = random.choice([0,1,2,3])
        # TF.rotate rotates counter clockwise, in degrees
        rotated_img = TF.rotate(img, 90 * random_rot_selection)
        assert rotated_img.shape == (3,64,64)

        cur_pos = action[0:2]
        cur_pos = np.array([cur_pos[1], cur_pos[0]]) # swap to get into x y form
        cur_pos -= 32 # get to centered at origin
        rotated_pos = rotate_around_origin_90s(cur_pos, random_rot_selection)
        rotated_pos = np.array([rotated_pos[1], rotated_pos[0]]) # unswap
        rotated_pos += 32
        assert rotated_pos.shape == (2,)

        rotated_action = np.array([sorted((0, rotated_pos[0], 63))[1], # clip it screw it
                                   sorted((0, rotated_pos[1], 63))[1], 
                                   (action[2] + random_rot_selection) % 2])

        # print(rotated_action)
        assert(rotated_action[0] < 64)
        assert(rotated_action[1] < 64)
        assert(rotated_action[2] < 2)
        return rotated_img, rotated_action
        ################################
        # Implement this function for Q4
        ################################
        return img, action

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = self.imgs[idx]
        action = self.actions[idx]

        H, W = img.shape[:2]
        img = TF.to_tensor(img)
        if np.random.rand() < 0.5:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        if self.train:
            img, action = self.transform_grasp(img, action)

        px, py, rot_id = action
        label = np.ravel_multi_index((rot_id, px, py), (2, H, W))

        return img, label

    def __len__(self) -> int:
        '''Number of grasps within dataset'''
        return self.imgs.shape[0]

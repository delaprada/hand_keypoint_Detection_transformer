import cv2
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import logging

torch.set_printoptions(edgeitems=48) # delete

logger = logging.getLogger(__name__)

class JointsDataset(Dataset):
  def __init__(self, cfg, root, video_set, is_train, transform=None):
    self.num_joints = 21
    self.pixel_std = 200
    self.flip_pairs = []
    self.parent_ids = []

    self.is_train = is_train
    self.root = root
    self.video_set = video_set

    self.output_path = cfg.OUTPUT_DIR
    self.data_format = cfg.DATASET.DATA_FORMAT

    self.scale_factor = cfg.DATASET.SCALE_FACTOR
    self.rotation_factor = cfg.DATASET.ROT_FACTOR
    self.flip = cfg.DATASET.FLIP
    self.color_rgb = cfg.DATASET.COLOR_RGB

    self.target_type = cfg.MODEL.TARGET_TYPE
    self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
    self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
    self.sigma = cfg.MODEL.SIGMA


    self.image_width = cfg.MODEL.IMAGE_SIZE[0]
    self.image_height = cfg.MODEL.IMAGE_SIZE[1]
    self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
    self.joints_weight = 1

    self.transform = transform
    self.aspect_ratio = self.image_width * 1.0 / self.image_height

    self.db = self._get_db()
    logger.info('=> load {} samples'.format(len(self.db)))

  def __len__(self,):
    return len(self.db)

  def _get_db(self):
    gt_db = self._load_keypoint_annotation()

    return gt_db
  
  def _load_keypoint_annotation(self):
    with open(self.root + 'annotations/hand_keypoints_' + self.video_set + '.json', 'r') as annotation:
      objs = json.load(annotation)
      rec = []

      for obj in objs:
        video_name = obj['name']
        frame_num = obj['frame_num']
        joints_3d = np.zeros((self.num_joints, 3), dtype=float)
        joints_3d_vis = np.zeros((self.num_joints, 3), dtype=float)

        # heatmap scale position
        joints_3d_ht = np.zeros((self.num_joints, 3), dtype=float)
        joints_3d_ht_vis = np.zeros((self.num_joints, 3), dtype=float)

        width = obj['width']
        height = obj['height']

        joints_3d_width, joints_3d_height = get_new_frame_size([width, height], self.image_size)
        joints_3d_ht_width, joints_3d_ht_height = get_new_frame_size([width, height], self.heatmap_size)

        label = obj['label']

        for i in range(len(label)):
          cur_label = label[i]

          for j in range(self.num_joints):
            joints_3d[j, 0] = cur_label[j * 2 + 1] * joints_3d_width
            joints_3d[j, 1] = cur_label[j * 2 + 2] * joints_3d_height
            joints_3d[j, 2] = 0

            # heatmap (heatmap 的大小和 image 的大小不同，所以需要生成对应的坐标，为后续生成 target joints 做准备)
            joints_3d_ht[j, 0] = cur_label[j * 2 + 1] * joints_3d_ht_width
            joints_3d_ht[j, 1] = cur_label[j * 2 + 2] * joints_3d_ht_height
            joints_3d_ht[j, 2] = 0

            t_vis = 1
            
            if cur_label[j * 2 + 1] > 1 or cur_label[j * 2 + 1] < 0 or cur_label[j * 2 + 2] > 1 or cur_label[j * 2 + 2] < 0:
              t_vis = 0
            
            joints_3d_vis[j, 0] = t_vis
            joints_3d_vis[j, 1] = t_vis
            joints_3d_vis[j, 2] = 0

            # heatmap
            joints_3d_ht_vis[j, 0] = t_vis
            joints_3d_ht_vis[j, 1] = t_vis
            joints_3d_ht_vis[j, 2] = 0

          rec.append({
            'video_path': self.root + 'videos/' + self.video_set + '/' + video_name,
            'joints_3d': copy.deepcopy(joints_3d),
            'joints_3d_vis': copy.deepcopy(joints_3d_vis),
            'joints_3d_ht': copy.deepcopy(joints_3d_ht),
            'joints_3d_ht_vis': copy.deepcopy(joints_3d_ht_vis),
            'frame_num': frame_num,
            'count': i
          })
      return rec

  def _load_video(self, path, max_frames=0):
    cap = cv2.VideoCapture(path)

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)
  
  def __getitem__(self, idx):
    db_rec = copy.deepcopy(self.db[idx])

    video_path = db_rec['video_path']
    joints = db_rec['joints_3d'].copy()
    joints_vis = db_rec['joints_3d_vis'].copy()

    # heatmap
    joints_ht = db_rec['joints_3d_ht']
    joints_ht_vis = db_rec['joints_3d_ht_vis']

    joints_heatmap = joints_ht.copy()

    count = db_rec['count'] # get current frame count
    frames = self._load_video(video_path + '.mp4')
    data_numpy = frames[count] # get current frame

    input = resize_img_keep_ratio(data_numpy, self.image_size)
    
    if self.transform:
      input = self.transform(input)

    target, target_weight = self.generate_target(joints_heatmap, joints_ht_vis)

    target = torch.from_numpy(target)
    target_weight = torch.from_numpy(target_weight)

    meta = {
      'video': video_path,
      'joints': joints.copy(),
      'joints_vis': joints_vis.copy(),
      'idx': idx,
    }

    return input, target, target_weight, meta
  
  # generate gt heatmap
  def generate_target(self, joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    assert self.target_type == 'gaussian', \
        'Only support gaussian map now!'

    if self.target_type == 'gaussian':
      target = np.zeros((self.num_joints,
                          self.heatmap_size[1],
                          self.heatmap_size[0]),
                        dtype=np.float32)

      for joint_id in range(self.num_joints):
        if target_weight[joint_id] == 0:
            continue
        
        # 关键点的 x, y 坐标
        mu_x = joints[joint_id][0]
        mu_y = joints[joint_id][1]
        
        x = np.arange(0, self.heatmap_size[0], 1, np.float32)
        y = np.arange(0, self.heatmap_size[1], 1, np.float32)
        y = y[:, np.newaxis]

        v = target_weight[joint_id]
        if v > 0.5:
          # 对每个关键点生成高斯分布（以下为高斯分布生成计算公式）
          target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

    if self.use_different_joints_weight:
      target_weight = np.multiply(target_weight, self.joints_weight)
  
    return target, target_weight

# get resized frame size
def get_new_frame_size(old_size, target_size):
  ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
  new_size = tuple([int(i*ratio) for i in old_size])
  return new_size

# resize frame and keep frame height width ratio
def resize_img_keep_ratio(frame, target_size):
    frame_size= frame.shape[0:2] # h, w
    old_size = [frame_size[1], frame_size[0]] # w, h

    # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))

    # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    new_size = tuple([int(i*ratio) for i in old_size])

    # 根据上边的大小进行放缩
    frame = cv2.resize(frame,(new_size[0], new_size[1]))

    # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_w = target_size[0] - new_size[0]

    # 计算需要填充的像素数目（图像的高这一维度上）
    pad_h = target_size[1] - new_size[1]

    # top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    # left, right = pad_w // 2, pad_w -(pad_w // 2)

    # 将空白部分填充在右边和下边，不影响 joints_3d 的坐标
    top, bottom = 0, pad_h
    left, right = 0, pad_w
    frame_new = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, None,(0,0,0))
    return frame_new

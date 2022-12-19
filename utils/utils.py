import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))

# get resized frame size
def get_new_frame_size(old_size, target_size):
  ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
  new_size = tuple([int(i*ratio) for i in old_size])

  pad_w = target_size[0] - new_size[0]
  gap = pad_w // 2

  return new_size, gap

# resize frame and keep frame height width ratio
def resize_img_keep_ratio(frame, target_size):
    frame_size= frame.shape[0:2] # h, w
    old_size = [frame_size[1], frame_size[0]] # w, h

    # get min ratio
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))

    # get new size of the frame according to the ratio
    new_size = tuple([int(i*ratio) for i in old_size])

    # resize frame according to new_size
    frame = cv2.resize(frame,(new_size[0], new_size[1]))

    # calculate padding on width scale
    pad_w = target_size[0] - new_size[0]

    # calculate padding on height scale
    pad_h = target_size[1] - new_size[1]

    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w -(pad_w // 2)

    frame_new = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, None,(0,0,0))

    return frame_new

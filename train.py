
import argparse
from config import cfg
from config import update_config
from core.function import train, validate
from core.loss import JointsMSELoss
import torch
import torchvision.transforms as transforms
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
import os
from tensorboardX import SummaryWriter

import models
import dataset

torch.cuda.set_device(4)
os.environ['CUDA_VISIBLE_DEVICES'] ='4'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def main():
  args = parse_args()
  update_config(cfg, args)

  logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

  logger.info(cfg)
  
  writer_dict = {
      'writer': SummaryWriter(log_dir=tb_log_dir),
      'train_global_steps': 0,
      'valid_global_steps': 0,
  }

  model = eval('models.' + cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True).cuda()
  model = torch.nn.DataParallel(model, device_ids=[4, 5]).cuda()

  criterion = JointsMSELoss(
    use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
  ).cuda()

  normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  )

  train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
    cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
    transforms.Compose([
      transforms.ToTensor(),
      normalize,
    ])
  )

  valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
  )

  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    shuffle=cfg.TRAIN.SHUFFLE,
    num_workers=cfg.WORKERS,
    pin_memory=cfg.PIN_MEMORY
  )

  valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    shuffle=False,
    num_workers=cfg.WORKERS,
    pin_memory=cfg.PIN_MEMORY
  )

  best_perf = 0.0
  best_model = False
  last_epoch = -1

  optimizer = get_optimizer(cfg, model)

  begin_epoch = cfg.TRAIN.BEGIN_EPOCH

  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)
  
  model.cuda()

  for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
    logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))

    # train for one epoch
    train(cfg, train_loader, model, criterion, optimizer, epoch, writer_dict)

    # validate for one epoch
    acc = validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir, writer_dict)

    lr_scheduler.step()

    if acc >= best_perf:
        best_perf = acc
        best_model = True
    else:
        best_model = False
    
    logger.info('=> saving checkpoint to {}'.format(final_output_dir))
    save_checkpoint({
        'epoch': epoch + 1,
        'model': cfg.MODEL.NAME,
        'state_dict': model.state_dict(),
        'best_state_dict': model.module.state_dict(),
        'perf': acc,
        'optimizer': optimizer.state_dict(),
        'train_global_steps': writer_dict['train_global_steps'],
        'valid_global_steps': writer_dict['valid_global_steps'],
    }, best_model, final_output_dir)

  final_model_state_file = os.path.join(
      final_output_dir, 'final_state.pth'
  )
  logger.info('=> saving final model state to {}'.format(
      final_model_state_file)
  )
  torch.save(model.module.state_dict(), final_model_state_file)
  writer_dict['writer'].close()

if __name__ == '__main__':
  main()

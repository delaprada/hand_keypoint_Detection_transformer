# 执行指令
```
python train.py --cfg ./experiments/TP_R_256x192_d256_h1024_enc4_mh8.yaml
```
或
```
python train.py --cfg ./experiments/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml
```

# ideas
- 采用先 easy mode -> 再 hard mode 的方式，先将 video 看作多张图片，跑通模型之后，再试着应用 vivit 里的方法
- 为什么不直接用 vivit 的方法：如果用 vivit 里的方法，transformer 要有 200 个输出（假设一个视频是 200 帧的话），每个输出有 21 个 channel(一只手有 21 个关键点)，每个 channel 对应一个关键点的 heatmap，每个 heatmap 都要和 ground-truth heatmap 计算 loss，过程有点复杂，所以先 try easy mode

# 2022.10.26 update
1. 去掉 affine transform（仿射变换）：先去掉仿射变换的原因是考虑到了以下几点
    - 关键点识别场景里没有像 coco 的 pose estimation 任务那样需要 bounding box 先将图片里的人物框出来，没有对应的 center, scale（也看不太懂源码是什么意思）
    - 仿射变换属于 data augmentation 的一种，一开始先不应用
2. 按照比例生成 joints_3d 和 joints_3d_ht
    - 仿射变换其中一个关键且不可或缺的作用是将 center, scale 代表 box 裁剪并缩放到 output_size，去掉仿射变换之后，我们需要另外的操作达到同样的效果。frame 会先被 resize 到 width 为 192, height 为 256 的大小，然后才作为后续 network 的 input。因此，joints_3d 和 joints_heatmap 也要按照比例变化才能准确标注到 frame 上关键点的位置。

      - 使用 get_new_frame_size 获取 frame 按比例缩小之后的 size 大小：
        ```python
        # 获取 resize 之后图片的大小，用于 resize joints_3d 和 joints_3d_ht
        def get_new_frame_size(old_size, target_size):
          ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
          new_size = tuple([int(i*ratio) for i in old_size])
          return new_size
        ```
      - 使用 resize_img_keep_ratio 将图片的长或宽不足 256 或 192 的边进行填充
        ```python
        # 封装resize函数
          def resize_img_keep_ratio(frame, target_size):
              frame_size= frame.shape[0:2] # h, w
              old_size = [frame_size[1], frame_size[0]] # w, h

              ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
              new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
              frame = cv2.resize(frame,(new_size[0], new_size[1])) # 根据上边的大小进行放缩
              pad_w = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的宽这一维度上）
              pad_h = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的高这一维度上）

              # top, bottom = pad_h // 2, pad_h - (pad_h // 2)
              # left, right = pad_w // 2, pad_w -(pad_w // 2)
              # 将空白部分填充在右边和下边，不影响 joints_3d 的坐标
              top, bottom = 0, pad_h
              left, right = 0, pad_w
              frame_new = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, None,(0,0,0))
            return frame_new
        ```
    - 根据新的 frame size 得到对应的 joints_3d 和 joint_3d_ht
      ```python
      joints_3d_width, joints_3d_height = get_new_frame_size([obj['width'], obj['height']], self.image_size)
        joints_3d_ht_width, joints_3d_ht_height = get_new_frame_size([obj['width'], obj['height']], self.heatmap_size)
      
      ......

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
      ......
      ```
3. 使用 copy.deepcopy 方法避免指向同样的内存地址：一开始训练的时候发现每张 frame 输出的关键点都是一样的异常现象，通过在 joints_3d 赋值时使用 copy.deepcopy 解决了这个问题。
```python
      for obj in objs:
        video_name = obj['name']
        frame_num = obj['frame_num']
        joints_3d = np.zeros((self.num_joints, 3), dtype=float)
        joints_3d_vis = np.zeros((self.num_joints, 3), dtype=float)

        # heatmap scale position
        joints_3d_ht = np.zeros((self.num_joints, 3), dtype=float)
        joints_3d_ht_vis = np.zeros((self.num_joints, 3), dtype=float)

        joints_3d_width, joints_3d_height = get_new_frame_size([obj['width'], obj['height']], self.image_size)
        joints_3d_ht_width, joints_3d_ht_height = get_new_frame_size([obj['width'], obj['height']], self.heatmap_size)

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
          
          width = obj['width']
          height = obj['height']

          x1 = np.max((0, width / 2))
          y1 = np.max((0, height / 2))
          x2 = np.min((width - 1, x1 + np.max((0, width - 1))))
          y2 = np.min((height - 1, y1 + np.max((0, height - 1))))
          if x2 >= x1 and y2 >= y1:
              obj['clean_bbox'] = [x1, y1, x2-x1+1, y2-y1+1]

          center, scale = self._box2cs(obj['clean_bbox'][:4])

          rec.append({
            'video_path': self.root + 'videos/' + self.video_set + '/' + video_name,
            'center': center,
            'scale': scale,
            'joints_3d': copy.deepcopy(joints_3d),
            'joints_3d_vis': copy.deepcopy(joints_3d_vis),
            'joints_3d_ht': copy.deepcopy(joints_3d_ht),
            'joints_3d_ht_vis': copy.deepcopy(joints_3d_ht_vis),
            'frame_num': frame_num,
            'count': i
          })
      return rec
```

import os
import cv2
import json
import numpy as np
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R
from dataset.base_dataset import BaseDataset


class oo3d9dsingle(BaseDataset):
    def __init__(self, data_path, data_name, data_type,
                 is_train=True, scale_size=480, num_view=50):
        super().__init__()

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = os.path.join(data_path, data_type)
        self.data_list = []
        with open(os.path.join(data_path, 'models_info_with_symmetry.json'), 'r') as f:
            self.models_info = json.load(f)
        with open(os.path.join(data_path, 'class_list.json'), 'r') as f:
            class_list = json.load(f)
            self.class_dict = {k: v for k, v in zip(class_list, range(len(class_list)))}
        for scene_id in os.listdir(self.data_path):
            with open(os.path.join(self.data_path, scene_id, 'scene_camera.json'), 'r') as f:
                scene_camera = json.load(f)
            with open(os.path.join(self.data_path, scene_id, 'scene_gt.json'), 'r') as f:
                scene_gt = json.load(f)
            with open(os.path.join(self.data_path, scene_id, 'scene_gt_info.json'), 'r') as f:
                scene_gt_info = json.load(f)
            with open(os.path.join(self.data_path, scene_id, 'scene_meta.json'), 'r') as f:
                scene_meta = json.load(f)
            curr_num_view = min(num_view, len(scene_camera.keys()))
            view_ids = np.array(list(scene_camera.keys()))
            # np.random.shuffle(view_ids)
            i = 0
            for view_id in view_ids:
                if i >= curr_num_view:
                    break
                if scene_gt_info[view_id][0]['bbox_visib'][2] < 50 or scene_gt_info[view_id][0]['bbox_visib'][3] < 50:
                    continue
                self.data_list.append(
                    {
                        'scene': scene_id,
                        'view': f'{int(view_id):{0}{6}}',
                        'cam': scene_camera[view_id],
                        'gt': scene_gt[view_id],
                        'gt_info': scene_gt_info[view_id],
                        'meta': scene_meta[view_id], 
                    }
                )
                i += 1
        
        phase = 'train' if is_train else 'test'
        print("Dataset: OmniObject3D Render")
        print("# of %s images: %d" % (phase, len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        info = self.data_list[idx]
        scene, view, cam, gt, gt_info, meta = info['scene'], info['view'], info['cam'], info['gt'], info['gt_info'], info['meta']
        num_obj = len(gt)
        # randomly select a sample
        visib_fract = [gt_info[i]['visib_fract'] for i in range(num_obj)]
        dist = (np.asarray(visib_fract) > 0.2).astype(float)
        if np.sum(dist) == 0:
            print(f'scene: {scene} | view: {view}')
            dist = (np.asarray(visib_fract) > 0).astype(float)
        dist /= dist.sum()
        sample_id = np.random.choice(np.arange(num_obj), p=dist)
        cam_K = np.asarray(cam['cam_K']).reshape(3, 3)
        cam_R_m2c, cam_t_m2c, obj_id = gt[sample_id]['cam_R_m2c'], gt[sample_id]['cam_t_m2c'], gt[sample_id]['obj_id']
        cam_R_m2c = np.asarray(cam_R_m2c).reshape(3, 3)
        cam_t_m2c = np.asarray(cam_t_m2c).reshape(1, 1, 3)
        kps3d = self.get_keypoints(meta[sample_id])
        diag = np.linalg.norm(kps3d[0, 1] - kps3d[0, 8])
        kp_i = (kps3d @ cam_R_m2c.T + cam_t_m2c) @ cam_K.T  # n * 9 * 3
        kp_i = kp_i[..., 0:2] / kp_i[..., 2:]  # n * 9 * 2

        bbox = gt_info[sample_id]['bbox_visib']

        rgb_path = os.path.join(self.data_path, scene, 'rgb', view+'.png')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.data_path, scene, 'rgb', view+'.jpg')
        depth_path = os.path.join(self.data_path, scene, 'depth', view+'.png')
        mask_path = os.path.join(self.data_path, scene, 'mask_visib', '_'.join([view, f'{sample_id:{0}{6}}'])+'.png')

        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        pcl_c = self.K_dpt2cld(depth, 1/cam['depth_scale'], cam_K)
        pcl_m = (pcl_c - cam_t_m2c).dot(cam_R_m2c)
        mask = cv2.imread(mask_path)
        image[mask != 255] = 70  # remove background

        if self.is_train:
            c, s = self.xywh2cs_dzi(bbox, wh_max=self.scale_size)
        else:
            c, s = self.xywh2cs(bbox, wh_max=self.scale_size)
        interpolate = cv2.INTER_NEAREST
        # interpolate = cv2.INTER_LINEAR
        rgb, c_h_, c_w_, s_ = self.zoom_in_v2(image, c, s, res=self.scale_size)
        pcl_m, *_ = self.zoom_in_v2(pcl_m, c, s, res=self.scale_size, interpolate=interpolate)
        mask, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=interpolate)
        mask = mask[..., 0] == 255
        center = (kps3d[0, 1] + kps3d[0, 8]) / 2
        nocs = (pcl_m - center.reshape(1, 1, 3)) / diag + 0.5
        nocs[np.logical_not(mask)] = 0
        mask[np.sum(np.logical_or(nocs > 1, nocs < 0), axis=-1) != 0] = False
        c = np.array([c_w_, c_h_])
        s = s_
        kp_i = (kp_i - c.reshape(1, 1, 2)) / s  # * self.scale_size
        
        dis_sym = np.zeros((3, 4, 4))
        if 'symmetries_discrete' in self.models_info[f'{obj_id}']:
            mats = np.asarray([np.asarray(mat_list).reshape(4, 4) for mat_list in self.models_info[f'{obj_id}']['symmetries_discrete']])
            dis_sym[:mats.shape[0]] = mats
        con_sym = np.zeros((3, 6))
        if 'symmetries_continuous' in self.models_info[f'{obj_id}']:
            for i, ao in enumerate(self.models_info[f'{obj_id}']['symmetries_continuous']):
                axis = np.asarray(ao['axis'])
                offset = np.asarray(ao['offset'])
                con_sym[i] = np.concatenate([axis, offset])

        if self.is_train:
            rgb = self.augment_training_data(rgb.astype(np.uint8))

        out_dict = {
            'image': (rgb.transpose((2, 0, 1)) / 255).astype(np.float32),
            'mask': mask,
            'nocs': nocs.transpose((2, 0, 1)).astype(np.float32), 
            'kps': kp_i.astype(np.float32),
            'dis_sym': dis_sym.astype(np.float32),
            'con_sym': con_sym.astype(np.float32), 
            'filename': '-'.join([scene, view]),
            'class_id': self.class_dict['_'.join(scene.split('_')[0:-2])],
            'obj_id': obj_id, 
        }

        if not self.is_train:
            kps3d = kps3d - center.reshape(1, 1, 3)
            pcl_c, *_ = self.zoom_in_v2(pcl_c, c, s, res=self.scale_size, interpolate=interpolate)
            extra_gt = {
                'dis_image': (rgb.transpose((2, 0, 1)) / 255).astype(np.float32),
                "kps3d": kps3d,
                "cam_K": cam_K,
                "cam_R_m2c": cam_R_m2c,
                "cam_t_m2c": cam_t_m2c,
                "c": c,
                "s": s,
                "diag": diag, 
                "pcl_c": pcl_c.astype(np.float32), 
            }
            out_dict.update(extra_gt)

        return out_dict

    @staticmethod
    def get_keypoints(model_info, dt=5):
        mins = [model_info['min_x'], model_info['min_y'], model_info['min_z']]
        sizes = [model_info['size_x'], model_info['size_y'], model_info['size_z']]
        maxs = [mins[i]+sizes[i] for i in range(len(mins))]
        base = [c.reshape(-1) for c in np.meshgrid(*zip(mins, maxs), indexing='ij')]
        base = np.stack(base, axis=-1)
        centroid = np.mean(base, axis=0, keepdims=True)
        base = np.concatenate([centroid, base], axis=0)
        keypoints = [base]
        if 'symmetries_discrete' in model_info:
            mats = [np.asarray(mat_list).reshape(4, 4) for mat_list in model_info['symmetries_discrete']]
            for mat in mats:
                curr = keypoints[0] @ mat[0:3, 0:3].T + mat[0:3, 3:].T
                keypoints.append(curr)
        elif 'symmetries_continuous' in model_info:
            # todo: consider multiple symmetries
            ao = model_info['symmetries_continuous'][0]
            axis = np.asarray(ao['axis'])
            offset = np.asarray(ao['offset'])
            angles = np.deg2rad(np.arange(dt, 180, dt))
            rotvecs = axis.reshape(1, 3) * angles.reshape(-1, 1)
            # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
            rots = R.from_rotvec(rotvecs).as_matrix()
            for rot in rots:
                curr = keypoints[0] @ rot.T + offset.reshape(1, 3)
                keypoints.append(curr)
        keypoints = np.stack(keypoints, axis=0)
        return keypoints

    @staticmethod
    def get_intr(h, w):
        fx = fy = 1422.222
        res_raw = 1024
        f_x = f_y = fx * h / res_raw
        K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        return K
    
    @staticmethod
    def read_camera_matrix_single(json_file):
        with open(json_file, 'r', encoding='utf8') as reader:
            json_content = json.load(reader)
        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = np.array(json_content['x'])
        camera_matrix[:3, 1] = -np.array(json_content['y'])
        camera_matrix[:3, 2] = -np.array(json_content['z'])
        camera_matrix[:3, 3] = np.array(json_content['origin'])

        c2w = camera_matrix
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        c2w = np.matmul(c2w, flip_yz)
        
        T_ = np.eye(4)
        T_[:3, :3] = R.from_euler('x', -90, degrees=True).as_matrix()
        c2w = np.matmul(T_, c2w)

        w2c = np.linalg.inv(c2w)

        return w2c[0:3, 0:3], w2c[0:3, 3].reshape(1, 1, 3) * 1000
    
    @staticmethod
    def K_dpt2cld(dpt, cam_scale, K):
        dpt = dpt.astype(np.float32)
        dpt /= cam_scale

        Kinv = np.linalg.inv(K)

        h, w = dpt.shape[0], dpt.shape[1]

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(w*h, 3)

        # backproj
        R = np.dot(Kinv, x2d.transpose())

        # compute 3D points
        X = R * np.tile(dpt.reshape(1, w*h), (3, 1))
        X = np.array(X).transpose()

        X = X.reshape(h, w, 3)
        return X
    
    @staticmethod
    def xywh2cs_dzi(xywh, base_ratio=1.5, sigma=1, shift_ratio=0.25, box_ratio=0.25, wh_max=480):
        # copy from
        # https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi/blob/master/lib/utils/img.py
        x, y, w, h = xywh
        shift = truncnorm.rvs(-shift_ratio / sigma, shift_ratio / sigma, scale=sigma, size=2)
        scale = 1+truncnorm.rvs(-box_ratio / sigma, box_ratio / sigma, scale=sigma, size=1)
        assert scale > 0
        center = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])])
        wh = max(w, h) * base_ratio * scale
        if wh_max != None:
            wh = min(wh, wh_max)
        return center, wh

    @staticmethod
    def xywh2cs(xywh, base_ratio=1.5, wh_max=480):
        x, y, w, h = xywh
        center = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        wh = max(w, h) * base_ratio
        if wh_max != None:
            wh = min(wh, wh_max)
        return center, wh
    
    @staticmethod
    def zoom_in_v2(im, c, s, res=480, interpolate=cv2.INTER_LINEAR):
        """
        copy from
        https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi/blob/master/lib/utils/img.py
        zoom in on the object with center c and size s, and resize to resolution res.
        :param im: nd.array, single-channel or 3-channel image
        :param c: (w, h), object center
        :param s: scalar, object size
        :param res: target resolution
        :param channel:
        :param interpolate:
        :return: zoomed object patch
        """
        c_w, c_h = c
        c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
        ndim = im.ndim
        if ndim == 2:
            im = im[..., np.newaxis]
        try:
            im_crop = np.zeros((s, s, im.shape[-1]))
        except:
            print(s)
            s = 480
            im_crop = np.zeros((s, s, im.shape[-1]))
        max_h, max_w = im.shape[0:2]
        crop_min_h, crop_min_w = max(0, c_h - s // 2), max(0, c_w - s // 2)
        crop_max_h, crop_max_w = min(max_h, c_h + s // 2), min(max_w, c_w + s // 2)
        up = s // 2 - (c_h - crop_min_h)
        down = s // 2 + (crop_max_h-c_h)
        left = s // 2 - (c_w - crop_min_w)
        right = s // 2 + (crop_max_w - c_w)
        im_crop[up:down, left:right] = im[crop_min_h:crop_max_h, crop_min_w:crop_max_w]
        im_crop = im_crop.squeeze()
        im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
        s = s
        if ndim == 2:
            im_resize = np.squeeze(im_resize)
        return im_resize, c_h, c_w, s

from tqdm import tqdm

import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.nn import SmoothL1Loss, L1Loss
from scipy.spatial.transform import Rotation as R
from pytorch3d.ops import box3d_overlap

from models.model import OV9D
from utils.aligning import estimateSimilarityTransform

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions


def oriented_box_to_axis_aligned_box(oriented_box):
    oriented_box = np.squeeze(oriented_box)
    mins = np.min(oriented_box, axis=0)
    maxs = np.max(oriented_box, axis=0)
    corners = [c.reshape(-1) for c in np.meshgrid(*zip(mins, maxs), indexing='ij')]
    corners = np.stack(corners, axis=-1)
    axis_aligned_box = [corners]
    return np.stack(axis_aligned_box, axis=0)


def main():
    opt = TestOptions()
    args = opt.initialize().parse_args()

    args.gpu = 'cuda:0'
    args.rank = 0
    device = torch.device(args.gpu)

    model = OV9D(args=args)
    cudnn.benchmark = True
    model.to(device)
    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model_weight.pop('encoder.class_embeddings')
    model.load_state_dict(model_weight, strict=False)
    model.eval()

    l1 = L1Loss()
    sl1 = SmoothL1Loss(beta=0.1)

    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset,
        'data_path': args.data_path,
        'data_name': args.data_name,
        'data_type': args.data_val,
        'num_view': 50, 
    }
    dataset_kwargs['scale_size'] = args.scale_size

    dataset = get_dataset(**dataset_kwargs, is_train=False)

    loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=1,
                                        shuffle=True, 
                                        num_workers=8, 
                                        pin_memory=True, 
                                        drop_last=False)

    err_t_list, err_r_list, err_pix_list = [], [], []
    l1_list, sl1_list = [], []
    iou_ob_list, iou_aab_list = [], []
    rel_iou_ob_list, rel_iou_aab_list = [], []
    T_mp2m_list = []
    box_list = []
    con_flag = False
    for batch_idx, batch in enumerate(tqdm(loader)):      
        input_RGB = batch['image'].to(device)
        # dis_RGB = batch['dis_image'].to(device)
        input_MASK = batch['mask'].to(device).to(bool)
        # 
        if torch.sum(input_MASK) < 100:
            continue
        nocs = batch['nocs'].to(device)
        dis_sym = batch['dis_sym'].to(device)
        con_sym = batch['con_sym'].to(device)
        if torch.sum(con_sym) != 0:
            sym_axis = con_sym[0, 0].detach().cpu().numpy().tolist().index(1.0)
            con_flag = True
        else:
            con_flag = False
        pcl_c = batch['pcl_c'].to(device)
        pcl_c = pcl_c[input_MASK].cpu().numpy()
        kps3d = batch['kps3d'][0].numpy()
        cam_R = batch['cam_R_m2c'][0].numpy()
        cam_t = batch['cam_t_m2c'][0].numpy()
        T = np.eye(4).astype(cam_R.dtype)  # T_m2mi
        T = torch.from_numpy(T).to(nocs)

        with torch.no_grad():
            preds = model(input_RGB, class_ids=batch['class_id'])
            pred_nocs = preds['pred_nocs'].permute(0, 2, 3, 1)[input_MASK]  # n 3
            # pred_nocs[..., 2] = 1 - pred_nocs[..., 2]  # !!!!!!!! for nocs real dataset only
            gt_nocs = nocs.permute(0, 2, 3, 1)[input_MASK]

            curr_pred_nocs = pred_nocs
            curr_gt_nocs = gt_nocs
            curr_pcl_m = curr_gt_nocs - 0.5  # nocs to pcl
            # discrete symmetry
            curr_dis_sym = dis_sym[0]
            dis_sym_flag = torch.sum(torch.abs(curr_dis_sym), dim=(1, 2)) != 0
            curr_dis_sym = curr_dis_sym[dis_sym_flag]
            aug_pcl_m = torch.stack([curr_pcl_m], dim=0)
            aug_T = torch.stack([T], dim=0)
            for sym in curr_dis_sym:
                rot, t = sym[0:3, 0:3], sym[0:3, 3]
                rot_pcl_m = aug_pcl_m @ rot.T + t.reshape(1, 1, 3)
                aug_pcl_m = torch.cat([aug_pcl_m, rot_pcl_m], dim=0)
                aug_T = torch.cat([aug_T, aug_T @ sym], dim=0)
            # continuous symmetry
            curr_con_sym = con_sym[0]
            con_sym_flag = torch.sum(torch.abs(curr_con_sym), dim=(-1)) != 0
            curr_con_sym = curr_con_sym[con_sym_flag]
            for sym in curr_con_sym:
                axis = sym[:3].cpu().numpy()
                angles = np.deg2rad(np.arange(5, 180, 5))
                rotvecs = axis.reshape(1, 3) * angles.reshape(-1, 1)
                rots = torch.from_numpy(R.from_rotvec(rotvecs).as_matrix()).to(curr_pcl_m)
                dTs = torch.from_numpy(np.stack([np.eye(4)] * rots.shape[0], axis=0)).to(curr_pcl_m)
                dTs[:, 0:3, 0:3] = rots
                rot_pcl_m_list = []
                aug_con_T_list = []
                for i in range(rots.shape[0]):
                    rot_pcl_m = aug_pcl_m @ rots[i].T
                    rot_pcl_m_list.append(rot_pcl_m)
                    aug_con_T_list.append(aug_T @ dTs[i])
                aug_pcl_m = torch.cat([aug_pcl_m] + rot_pcl_m_list, dim=0)
                aug_T = torch.cat([aug_T] + aug_con_T_list, dim=0)
            curr_gt_nocs_set = aug_pcl_m + 0.5
            curr_gt_nocs_set = torch.unbind(curr_gt_nocs_set, dim=0)
            loss = list(map(lambda nocs: sl1(curr_pred_nocs, nocs), curr_gt_nocs_set))
            min_idx = torch.argmin(torch.tensor(loss))
            gt_nocs = curr_gt_nocs_set[min_idx]
            T_mp2m_sym = aug_T[min_idx]
            sl1_list.append(sl1(pred_nocs, gt_nocs).detach().cpu().numpy())
            l1_list.append(l1(pred_nocs, gt_nocs).detach().cpu().numpy())
        
        pred_nocs = pred_nocs - 0.5
        pred_coord_pts = pred_nocs
        pred_coord_pts = pred_coord_pts.detach().cpu().numpy()
        gt_nocs = gt_nocs - 0.5
        gt_coord_pts = gt_nocs
        gt_coord_pts = gt_coord_pts.detach().cpu().numpy()

        pred_scales, pred_rotation, pred_translation, _ = estimateSimilarityTransform(pred_coord_pts, pcl_c, False)
        gt_scales, gt_rotation, gt_translation, _ = estimateSimilarityTransform(gt_coord_pts, pcl_c, False)
        T_m2c = np.eye(4)
        T_m2c[0:3, 0:3], T_m2c[0:3, 3] = cam_R, np.squeeze(cam_t)
        # T_m2c[0:3, 0:3], T_m2c[0:3, 3] = gt_rotation.T, np.squeeze(gt_translation)
        T_c2m = np.linalg.inv(T_m2c)
        T_mp2m_sym = T_mp2m_sym.cpu().numpy()
        T_m_sym2c = np.eye(4)
        T_m_sym2c[0:3, 0:3], T_m_sym2c[0:3, 3] = pred_rotation.T, pred_translation
        T_mp2c = T_m_sym2c @ T_mp2m_sym
        T_mp2m = T_c2m @ T_mp2c
        T_mp2m_list.append(T_mp2m)

        err_t = np.linalg.norm(pred_translation - gt_translation) / 1000
        if con_flag:
            err_r = np.rad2deg(np.arccos(max(-1, min(1, pred_rotation.T[1] @ gt_rotation.T[1]))))
        else:
            err_r = np.rad2deg(np.linalg.norm(R.from_matrix(pred_rotation @ gt_rotation.T).as_rotvec()))
        err_t_list.append(err_t)
        err_r_list.append(err_r)

        vids = np.array([1,5,7,3,2,6,8,4])
        pytorch3d_box = kps3d[0:1, vids]
        box_list.append(pytorch3d_box)
        box1 = pytorch3d_box @ pred_rotation + pred_translation
        box2 = pytorch3d_box @ gt_rotation + gt_translation
        intersection_vol, iou_ob = box3d_overlap(torch.from_numpy(box1).to(torch.float32), 
                                                 torch.from_numpy(box2).to(torch.float32), 
                                                 eps=2e-3)
        vids = vids - 1
        aab1 = oriented_box_to_axis_aligned_box(box1)[:, vids]
        aab2 = oriented_box_to_axis_aligned_box(box2)[:, vids]
        intersection_vol, iou_aab = box3d_overlap(torch.from_numpy(aab1).to(torch.float32), 
                                                  torch.from_numpy(aab2).to(torch.float32), 
                                                  eps=2e-3)
        
        iou_ob_list.append(iou_ob.numpy()[0,0])
        iou_aab_list.append(iou_aab.numpy()[0,0])
    
    print(f'mean l1: {np.mean(l1_list)}')
    print(f'mean sl1: {np.mean(sl1_list)}')

    iou_ob = np.asarray(iou_ob_list)
    iou_aab = np.asarray(iou_aab_list)

    err_t = np.asarray(err_t_list)
    err_r = np.asarray(err_r_list)
    # err_pix = np.asarray(err_pix_list)
    avg_err_t = np.mean(err_t)
    avg_err_r = np.mean(err_r)
    # avg_err_pix = np.mean(err_pix)
    t2r2 = np.sum(np.logical_and(err_t < 0.02, err_r < 2)) / err_t.shape[0]
    t2r5 = np.sum(np.logical_and(err_t < 0.02, err_r < 5)) / err_t.shape[0]
    t5r5 = np.sum(np.logical_and(err_t < 0.05, err_r < 5)) / err_t.shape[0]
    t5r10 = np.sum(np.logical_and(err_t < 0.05, err_r < 10)) / err_t.shape[0]
    t10r10 = np.sum(np.logical_and(err_t < 0.1, err_r < 10)) / err_t.shape[0]
    iou_ob_25 = np.sum(iou_ob > 0.25) / iou_ob.shape[0]
    iou_ob_50 = np.sum(iou_ob > 0.5) / iou_ob.shape[0]
    iou_ob_75 = np.sum(iou_ob > 0.75) / iou_ob.shape[0]
    iou_aab_25 = np.sum(iou_aab > 0.25) / iou_aab.shape[0]
    iou_aab_50 = np.sum(iou_aab > 0.5) / iou_aab.shape[0]
    iou_aab_75 = np.sum(iou_aab > 0.75) / iou_aab.shape[0]

    print(f'avg translation difference: {avg_err_t} m')
    print(f'avg rotation difference: {avg_err_r} deg')
    print(f'2 deg 2 cm: {t2r2}')
    print(f'5 deg 2 cm: {t2r5}')
    print(f'5 deg 5 cm: {t5r5}')
    print(f'10 deg 5 cm: {t5r10}')
    print(f'10 deg 10 cm: {t10r10}')
    print(f'IoU-OB-3D@25: {iou_ob_25}')
    print(f'IoU-OB-3D@50: {iou_ob_50}')
    print(f'IoU-OB-3D@75: {iou_ob_75}')
    print(f'IoU-AAB-3D@25: {iou_aab_25}')
    print(f'IoU-AAB-3D@50: {iou_aab_50}')
    print(f'IoU-AAB-3D@75: {iou_aab_75}')
    
    T_mp2m = np.stack(T_mp2m_list, axis=0)
    R_mp2m = T_mp2m[:, 0:3, 0:3]
    if con_flag:
        # R_m2mp = R_mp2m.transpose((0, 2, 1))
        DR = R_mp2m[None] @ R_mp2m.transpose((0, 2, 1))[:, None]
        err_r_mat = np.rad2deg(np.arccos(np.clip(DR[:, :, sym_axis, sym_axis], -1, 1)))
    else:
        DR = R_mp2m.transpose((0, 2, 1))[None] @ R_mp2m[:, None]
        err_r_mat = np.rad2deg(np.linalg.norm(R.from_matrix(DR.reshape(-1, 3, 3)).as_rotvec().reshape(DR.shape[0], DR.shape[0], 3), axis=-1))
    idx = np.argmin(np.sum(err_r_mat, axis=1))
    err_r = err_r_mat[idx]
    t_mp2m = T_mp2m[:, 0:3, 3]
    err_t_mat = np.linalg.norm(t_mp2m[None] - t_mp2m[:, None], axis=-1)
    err_t = err_t_mat[idx] / 1000

    T_mi2m = T_mp2m[idx]
    T_m2mi = np.linalg.inv(T_mi2m)
    
    for j in range(err_t.shape[-1]):
        bbox_m = box_list[j]
        bbox_i = bbox_m @ T_m2mi[0:3, 0:3].T + T_m2mi[0:3, 3].reshape(1, 1, 3)
        T_m2mj = np.linalg.inv(T_mp2m[j])
        T_mi2mj = T_m2mj @ T_mi2m
        bbox_j = bbox_i @ T_mi2mj[0:3, 0:3].T + T_mi2mj[0:3, 3].reshape(1, 1, 3)
        intersection_vol, iou_ob = box3d_overlap(torch.from_numpy(bbox_i).to(torch.float32), 
                                                 torch.from_numpy(bbox_j).to(torch.float32), 
                                                 eps=1e-3)
        vids = np.array([0,4,6,2,1,5,7,3])
        aab1 = oriented_box_to_axis_aligned_box(bbox_i)[:, vids]
        aab2 = oriented_box_to_axis_aligned_box(bbox_j)[:, vids]
        intersection_vol, iou_aab = box3d_overlap(torch.from_numpy(aab1).to(torch.float32), 
                                                  torch.from_numpy(aab2).to(torch.float32), 
                                                  eps=1e-3)
        
        rel_iou_ob_list.append(iou_ob.numpy()[0,0])
        rel_iou_aab_list.append(iou_aab.numpy()[0,0])

    rel_iou_ob = np.asarray(rel_iou_ob_list)
    rel_iou_aab = np.asarray(rel_iou_aab_list)

    avg_err_t = np.mean(err_t)
    avg_err_r = np.mean(err_r)
    # avg_err_pix = np.mean(err_pix)

    t2r2 = np.sum(np.logical_and(err_t < 0.02, err_r < 2)) / err_t.shape[0]
    t2r5 = np.sum(np.logical_and(err_t < 0.02, err_r < 5)) / err_t.shape[0]
    t5r5 = np.sum(np.logical_and(err_t < 0.05, err_r < 5)) / err_t.shape[0]
    t5r10 = np.sum(np.logical_and(err_t < 0.05, err_r < 10)) / err_t.shape[0]
    t10r10 = np.sum(np.logical_and(err_t < 0.1, err_r < 10)) / err_t.shape[0]
    rel_iou_ob_25 = np.sum(rel_iou_ob > 0.25) / rel_iou_ob.shape[0]
    rel_iou_ob_50 = np.sum(rel_iou_ob > 0.5) / rel_iou_ob.shape[0]
    rel_iou_ob_75 = np.sum(rel_iou_ob > 0.75) / rel_iou_ob.shape[0]
    rel_iou_aab_25 = np.sum(rel_iou_aab > 0.25) / rel_iou_aab.shape[0]
    rel_iou_aab_50 = np.sum(rel_iou_aab > 0.5) / rel_iou_aab.shape[0]
    rel_iou_aab_75 = np.sum(rel_iou_aab > 0.75) / rel_iou_aab.shape[0]

    print(f'avg rel translation difference: {avg_err_t} m')
    print(f'avg rel rotation difference: {avg_err_r} deg')
    print(f'rel 2 deg 2 cm: {t2r2}')
    print(f'rel 5 deg 2 cm: {t2r5}')
    print(f'rel 5 deg 5 cm: {t5r5}')
    print(f'rel 10 deg 5 cm: {t5r10}')
    print(f'rel 10 deg 10 cm: {t10r10}')
    print(f'IoU-OB-3D@25: {rel_iou_ob_25}')
    print(f'IoU-OB-3D@50: {rel_iou_ob_50}')
    print(f'IoU-OB-3D@75: {rel_iou_ob_75}')
    print(f'IoU-AAB-3D@25: {rel_iou_aab_25}')
    print(f'IoU-AAB-3D@50: {rel_iou_aab_50}')
    print(f'IoU-AAB-3D@75: {rel_iou_aab_75}')


if __name__ == '__main__':
    main()

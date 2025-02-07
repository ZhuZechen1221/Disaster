import json
import os
import numpy as np
import cv2
import pickle
from smpl_np import SMPLModel
from DisasterLoader import DisasterDatasetLoader
from scipy.spatial.transform import Rotation as R
import pandas as pd
from collections import defaultdict
import math
FOV = 65.579994  #degree
SENSOR_WIDTH = 24.576 #mm
Folcal_Length = 19.074556 #mm
Scale_Factor = 0.0192
import torch

# rootpath = "./DisasterSet/dataset_1file"
rootpath = "./DisasterSet/dataset_4file"
# rootpath = "./DisasterSet/testset_79file"
# rootpath = "./DisasterSet/disasterset_1000"
IMAGE_IDS = 10000
ANNOTATION_IDS = 20000

joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )

UVJointOrder = ('Head', 'L_Ankle', 'L_Collar', 'L_Elbow', 
                'L_Foot', 'L_Hand', 'L_Hip', 'L_Knee', 
                'L_Shoulder', 'L_Wrist', 'Neck', 'Pelvis', 
                'R_Ankle', 'R_Collar', 'R_Elbow', 'R_Foot', 
                'R_Hand', 'R_Hip', 'R_Knee', 'R_Shoulder', 
                'R_Wrist', 'Spine1', 'Spine2', 'Spine3', 
                'root')

SMPLJoint24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb')             # 23

GlobalJointOrder = (
        'SMPL-male_001', 'root', 
        'pelvis', 'left_hip', 'left_knee', 
        'left_ankle', 'left_foot', 'right_hip', 
        'right_knee', 'right_ankle', 'right_foot', 
        'Spine1', 'Spine2', 'Spine3', 
        'neck', 'jaw', 
        'left_collar', 'left_shoulder', 'left_elbow', 
        'left_wrist', 'left_thumb', 'right_collar', 
        'right_shoulder', 'right_elbow', 'right_wrist', 'right_thumb')

def rotation_matrix_to_axis_angle_vector(R):
    # 计算旋转角度 theta
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    # 如果 theta 非零，计算旋转轴
    if np.isclose(theta, 0):
        axis = np.array([0, 0, 0])  # 无旋转
    else:
        vx = (R[3, 2] - R[2, 3]) / (2 * np.sin(theta))
        vy = (R[1, 3] - R[3, 1]) / (2 * np.sin(theta))
        vz = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        axis = np.array([vx, vy, vz])
    
    # 轴角表示：轴乘以角度
    axis_angle = theta * axis
    return axis_angle

def rotation_matrix_to_axis_angle_vector_torch(R_a):
    # calculate rotation angle
    R = torch.tensor(R_a, dtype=torch.double).cuda()
    theta = torch.arccos((torch.trace(R) - 1) / 2)
    zero = torch.tensor(0, dtype=torch.double).cuda()
    # theta is not zero
    if torch.isclose(theta, zero):
        axis = torch.tensor([0, 0, 0])  # without rotation
    else:
        vx = (R[2, 1] - R[1, 2]) / (2 * torch.sin(theta))
        vy = (R[0, 2] - R[2, 0]) / (2 * torch.sin(theta))
        vz = (R[1, 0] - R[0, 1]) / (2 * torch.sin(theta))
        axis = torch.tensor([vx, vy, vz]).cuda()
    
    # axis angle in 3 element: angle * axis
    axis_angle = theta * axis
    axis_angle = axis_angle.cpu().numpy()
    return axis_angle

def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(r.dtype).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
        z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
        r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
        -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
        np.expand_dims(np.eye(3), axis=0),
        [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

def save_img_with_bb(bbs, img_paths, mask_paths):
    '''
    draw binding box on image and mask, saved into filefolder
    '''
    for i in range(len(bbs)): #i for sequence index
        for j in range(len(bbs[i])):   # j for image/mask/bb index
            x, y, w, h = bbs[i][j][0], bbs[i][j][1], bbs[i][j][2], bbs[i][j][3]
            x0, y0 = x - 20, y - 20
            x1, y1 = x0 + w + 20, y0 + w + 20
            # draw rectangle on image
            image = cv2.imread(img_paths[i][j])
            ps = img_paths[i][j].split('/')  #:./DisasterSet/dataset/Sequence/image/maskname
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 3)
            savepth = os.path.join(ps[0], ps[1], 'bindingbox', ps[3],'images', ps[-1])
            success = cv2.imwrite(savepth, image)
            if success: print(f"saved to : {savepth}")
            else: print("failed")
            #draw rectangle on mask
            mask = cv2.imread(mask_paths[i][j])
            ps = mask_paths[i][j].split('/')  #:./DisasterSet/dataset/Sequence/image/maskname
            cv2.rectangle(mask, (x0, y0), (x1, y1), (0, 0, 255), 3)
            savepth = os.path.join(ps[0], ps[1], 'bindingbox', ps[3], 'masks', ps[-1])
            success = cv2.imwrite(savepth, mask)
            if success: print(f"saved to : {savepth}")
            else: print("failed")
    return 

def cal_joint_cam(cam_loc,cam_rot,joints_world):
    joints_cam = []
    for joint in joints_world:
        joint_cam = cam_rot @ joint + cam_loc
        joints_cam.append(joint_cam)
    return joints_cam

def cal_joint_cam_torch(cam_loc,cam_rot,joints_world):
    joints_cam = []
    for joint in joints_world:
        joint_torch = torch.tensor(joint, dtype=torch.double).cuda()
        cam_rot_troch = torch.tensor(cam_rot, dtype=torch.double).cuda()
        cam_loc_torch = torch.tensor(cam_loc, dtype=torch.double).cuda()
        joint_cam_torch = cam_rot_troch @ joint_torch + cam_loc_torch
        joint_cam = joint_cam_torch.cpu().numpy()
        joints_cam.append(joint_cam)
    return joints_cam

def cal_uvd_joint(joint_cam, joint_uv):
    uvd_joints = []
    for joint, uv in zip(joint_cam, joint_uv):
        depth = (joint[2] - Folcal_Length)
        uvd_joints.append([uv[0], uv[1], depth])  # pixel, pixel, mm
    return uvd_joints

def convert_to(data):
    if isinstance(data, list):  # 如果是列表，递归处理每个元素
        return [convert_to(item) for item in data]
    elif isinstance(data, np.int64):  # 如果是 int64 或 float64，转换为 int
        return int(data)
    elif isinstance(data, np.float64):  # 如果是 int64 或 float64，转换为 int
        return float(data)
    elif isinstance(data, np.bool_):  # 如果是 bool_，转换为 bool
        return bool(data)
    else:  # 其他数据类型保持不变
        return data

def make_json(disasterset):
    '''get values from disaster dataset'''
    sequences = disasterset.sequences
    img_names = disasterset.img_names
    img_paths = disasterset.img_paths
    hw_princpts = disasterset.hw_princpts  #pixel
    focal_length = disasterset.focal_length  #19.074556 mm   about 1000 pixel
    cam_locs = disasterset.cam_locs   #cam xyz     #sequences-files-camera-metrix 
    cam_rots = disasterset.cam_rots   #cam rot mat   #sequences-files-camera-metrix
    joint_visibles = disasterset.joint_visibles
    uv_joints = disasterset.uv_joints   #pixel
    betas = disasterset.betas
    joint_locs_world = disasterset.joint_locs_world   #sequence-file-joint
    rot_mats = disasterset.rot_mats
    genders_trans = disasterset.genders_trans
    bboxs = disasterset.bboxs
    '''all image name and sequences'''
    print('create image name...')
    image_name = []
    for sequence_ in img_names:
        for name in sequence_:
            image_name.append(name)
    '''all sequence name'''
    nums = []
    print('create sequence name...')
    for sequence_ in img_names:
        nums.append(len(sequence_)) 

    sub_sequences = [] 
    for i in range(len(nums)):
        sub_sequence = [sequences[i]] * nums[i]
        sub_sequences.append(sub_sequence)

    sequence = []
    for sequence_ in sub_sequences:
        for item in sequence_:
            sequence.append(item)
    focal = []
    for s in img_paths:
        s_index = 0
        for i in range(len(s)): 
            focal.append([focal_length, focal_length])
        s_index += 1
    '''index for all images and annotations'''
    print('create sequence index...')
    image_id = np.random.randint(10000, 20000, size=len(image_name))
    ann_id = np.random.randint(20000, 30000, size=len(image_name))
    image_id = image_id.tolist()
    ann_id = ann_id.tolist()
    '''width and height and princpt for all images'''
    print('create w h and princpt...')
    width = []
    height = []
    princpt = []
    for s in hw_princpts:
        for hw_p in s:
            width.append(hw_p[0])
            height.append(hw_p[1])
            princpt.append(hw_p[2])
    '''joint location in cam'''
    print('create joint in cam...')
    smpl_joint_cams = []
    cam_loc = []
    cam_rot = []
    for s1, s2 in zip(cam_locs, cam_rots):
        for locs in s1: cam_loc.append(locs)
        for rots in s2: cam_rot.append(rots)
    print('finished cam_loc and cam_rot')
    start = 0
    for sequence_ in joint_locs_world:
        for num in nums:
            if start <= len(cam_loc) -1:
                for cam_num in range(start, start+num):
                    print('calculate',start,'-th for joint cam')
                    joints_cam = cal_joint_cam_torch(cam_loc[cam_num], cam_rot[cam_num], sequence_)
                    smpl_joint_cams.append(joints_cam)   
            start += num         
    '''joint location in uvd format'''
    print('create uvd joint')
    smpl_joints_imgs = []
    flag = 0
    for joint_cam, joint_uv in zip(smpl_joint_cams, uv_joints):
        flag += 1
        print('calculate',flag,'-th for joint uvd')
        smpl_joints_img = cal_uvd_joint(joint_cam, joint_uv)
        smpl_joints_imgs.append(smpl_joints_img)
    '''shape and pose parameter, gender and tran'''
    print('create shape, pose, gender, tran...')
    beta = []
    theta = []
    genders = []
    trans = []
    flag = 0
    for num in nums:
        print(num)
        for i in range(num):
            beta.append(betas[flag])
            theta.append(rot_mats[flag])
            genders.append(genders_trans[flag][0][0][5:])
            trans.append([genders_trans[flag][0][1], genders_trans[flag][0][2], genders_trans[flag][0][3]])
        flag += 1
    '''binding box in xyxy'''
    print('create bbs')
    bbs = []
    for sequence_ in bboxs:
        for bb in sequence_:
            x = int(bb[0])
            y = int(bb[1])
            w = int(bb[2])
            h = int(bb[3])
            bbs.append([x, y, w, h])
    '''create joint 17 in coco'''
    print('create joint 17...')
    smpl_joint_17 = []
    for i in smpl_joint_cams:
        joint = [None] * 17
        joint[0] = i[0]
        joint[1] = i[1]
        joint[2] = i[4]
        joint[3] = i[7]
        joint[4] = i[2]
        joint[5] = i[5]
        joint[6] = i[8]
        joint[7] = i[3]
        joint[8] = i[12]
        joint[9] = i[15]    #nose and head are same, corresponding to jaw in smpl
        joint[10] = i[15]
        joint[11] = i[16]
        joint[12] = i[18]
        joint[13] = i[20]
        joint[14] = i[17]
        joint[15] = i[19]
        joint[16] = i[21]
        smpl_joint_17.append(joint)
    '''visible of joint 17'''
    print('create visible 17...')
    smpl_vivible_17 = []
    for viss in joint_visibles:
        viss_01 = [None] * 17
        viss_01[0] = 1 if viss[0][1] else 0
        viss_01[1] = 1 if viss[1][1] else 0
        viss_01[2] = 1 if viss[4][1] else 0
        viss_01[3] = 1 if viss[7][1] else 0
        viss_01[4] = 1 if viss[2][1] else 0
        viss_01[5] = 1 if viss[5][1] else 0
        viss_01[6] = 1 if viss[8][1] else 0
        viss_01[7] = 1 if viss[3][1] else 0
        viss_01[8] = 1 if viss[12][1] else 0
        viss_01[9] = 1 if viss[15][1] else 0
        viss_01[10] = 1 if viss[15][1] else 0
        viss_01[11] = 1 if viss[16][1] else 0
        viss_01[12] = 1 if viss[18][1] else 0
        viss_01[13] = 1 if viss[20][1] else 0
        viss_01[14] = 1 if viss[17][1] else 0
        viss_01[15] = 1 if viss[19][1] else 0
        viss_01[16] = 1 if viss[21][1] else 0
        smpl_vivible_17.append(viss_01)
    '''transform theta in to axis angle'''
    print('create rot anix...')
    flag = 0
    theta_axis = []
    for item in theta:
        axis_joint = []
        for R in item:
            flag += 1
            print('calculate',flag,'-th for joint axis')
            axis_angle = rotation_matrix_to_axis_angle_vector_torch(R)
            axis_angle = list(axis_angle)
            axis_joint.append(axis_angle)
        theta_axis.append(axis_joint)
    '''transform data shapes'''
    print('transform data shape...')
    theta_axis = np.array(theta_axis).reshape((len(theta_axis), 72))  
    theta_axis = theta_axis.tolist()
    smpl_joint_17 = np.array(smpl_joint_17).reshape((len(smpl_joint_17), 51))
    smpl_joint_17 = smpl_joint_17.tolist()
    smpl_joint_cams = np.array(smpl_joint_cams).reshape((len(smpl_joint_cams), 72))
    smpl_joint_cams = smpl_joint_cams.tolist()
    '''visible of joint 24'''
    print('create joint 24 visible...')
    smpl_vivible_24 = []
    for viss in joint_visibles:
        joint_viss = []
        for vis in viss:
            if vis[0] != 'root':
                if vis[1] is True: joint_viss.append(1)
                else: joint_viss.append(0)
            else: pass
        smpl_vivible_24.append(joint_viss)
    '''make json dict'''  
    print('create json dict...')
    images = []
    anns = []
    for i in range(len(image_id)):
        img = {
                'id': image_id[i],
                'file_name': image_name[i],
                'sequence': sequence[i],
                'width': width[i],
                'height': height[i],
                'cam_param': {'focal': focal[i], 'princpt': princpt[i]}}
        ann =  {
                'id': ann_id[i],
                'image_id': image_id[i],
                'fitted_3d_pose': smpl_joints_imgs[i],
                'smpl_param': {'shape': beta[i], 'pose': theta_axis[i], 'trans': trans[i], 'gender': genders[i]},
                'bbox': bbs[i],
                'h36m_joints': smpl_joint_17[i],
                'smpl_joint_img': smpl_joints_imgs[i],
                'smpl_joint_cam': smpl_joint_cams[i],
                'smpl_24_visible': smpl_vivible_24[i],
                'h36m_joints_visible': smpl_vivible_17[i]}
        images.append(img)
        anns.append(ann)
    json_dict = {'images': images, 'annotations': anns}    

    #print(len(image_id), len(image_name), len(sequence), len(width), len(height), len(height), len(focal), len(princpt))
    #print(len(ann_id), len(smpl_joints_imgs), len(beta), len(theta_axis), len(trans), len(genders), len(smpl_joint_17), len(smpl_joints_imgs), len(smpl_joint_cams), len(smpl_vivible_24), len(smpl_vivible_17))


    return json_dict

def save_json(json_dict):
    print('saving json file...')
    file_name = rootpath.split('/')[-1] + '.json'
    pth = os.path.join('./', file_name)
    with open(pth, "w") as file:   # re-save json
        json.dump(json_dict, file, indent=4)
    print('finished')
    return 

disasterset = DisasterDatasetLoader(rootpath=rootpath)

json_dict = make_json(disasterset)
save_json(json_dict)

def rotation_matrix_to_ypr(R):
    """
    从旋转矩阵 R 提取 Yaw-Pitch-Roll (右手系，顺序 ZYX)
    :param R: 3x3 旋转矩阵
    :return: (yaw, pitch, roll) 角度（弧度制）
    """
    # 提取 pitch (theta)
    pitch = np.arcsin(-R[2, 0])
    
    # 检查是否接近奇异点
    if np.abs(pitch - np.pi / 2) < 1e-6:
        # 奇异点：theta = +90°
        yaw = 0
        roll = np.arctan2(R[0, 1], R[0, 2])
    elif np.abs(pitch + np.pi / 2) < 1e-6:
        # 奇异点：theta = -90°
        yaw = 0
        roll = -np.arctan2(R[0, 1], R[0, 2])
    else:
        # 正常情况
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    
    return yaw, pitch, roll


def run_disaster_GT():
    file_name = rootpath.split('/')[-1]+'.json'
    pth = os.path.join(rootpath, file_name)
    with open(pth, 'r', encoding='utf-8') as file:
        data = json.load(file)
    images = data['images']
    anns= data['annotations']
    image = images[0]
    ann = anns[0]
    pose = ann['smpl_param']['pose']
    shape = ann['smpl_param']['shape']
    tran = ann['smpl_param']['trans']
    save_name = image['file_name'][:-4]

    pose = np.array(pose).reshape((24,1,3))
    pose_R = rodrigues(pose)
    shape = np.array(shape)
    tran = np.array(tran)
    smpl = SMPLModel('./DisasterSet/SMPL_NEUTRAL.pkl')
    smpl_name = './DisasterSet/smpl_saves/test_ypr/' + save_name + '_joint0.obj'
    smpl.set_params(beta = shape, posemat = pose_R, trans = tran)
    smpl.save_to_obj(smpl_name)
    return


def run_3dpw_GT():
    filename ='./data/pw3d/json/3DPW_test_new.json'
    with open(filename, "r", encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    images = data['images']
    anns= data['annotations']
    image = images[0]
    ann = anns[0]
    pose = ann['smpl_param']['pose']
    shape = ann['smpl_param']['shape']
    tran = ann['smpl_param']['trans']
    save_name = image['file_name'][:-4]

    pose = np.array(pose).reshape((24,1,3))
    pose_R = rodrigues(pose)
    # for R in pose_R:
    #     yaw, pitch, roll = rotation_matrix_to_ypr(R)

    shape = np.array(shape)
    tran = np.array(tran)

    # smpl = SMPLModel('./DisasterSet/SMPL_NEUTRAL.pkl')
    # smpl_name = './DisasterSet/smpl_saves/' + save_name + '.obj'
    # smpl.set_params(beta = shape, posemat = pose_R, trans = tran)
    # smpl.save_to_obj(smpl_name)

# run_3dpw_GT()






# pth = './propose_disaster_test.pkl'
# with open(pth,'rb') as f:
#     data = pickle.load(f)

# data = data[-2][1]
# pose = data['pred_theta_mats'].cpu().numpy().reshape((24,3,3))

# ypr_test = []
# for R in pose:
#     yaw, pitch, roll = rotation_matrix_to_ypr(R)
#     ypr_test.append([math.degrees(yaw), math.degrees(pitch), math.degrees(roll)])
# # print(ypr_test)

# for i in ypr_test:
#     print(i)



def compute_mpjpe(predicted, ground_truth):
    """
    :param predicted joint position (24, 3)
    :param ground_truth joint position (24, 3)
    :return: MPJPE (float)
    """
    assert predicted.shape == ground_truth.shape
    
    # 计算每个关节的欧几里得距离
    errors = np.linalg.norm(predicted - ground_truth, axis=1)
    
    # 返回平均误差
    return np.mean(errors)




# python scripts/demo.py --img-dir ./examples --out-dir dump_demo --ckpt './model_files/propose_hr48_xyz.pth'
#  sh ./scripts/train.sh disaster1000_test1 ./configs/smpl_hm_xyz.yaml

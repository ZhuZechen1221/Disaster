import os
from PIL import Image
import math
import pandas as pd
import numpy as np
import cv2

FOV = 65.579994  #degree
SENSOR_WIDTH = 24.576 #mm
Folcal_Length = 19.074556 #mm
Scale_Factor = 0.0192
rootpath = "./DisasterSet/dataset"
IMAGE_IDS = 10000
ANNOTATION_IDS = 20000
L_Hand = 24
R_Hand = 25
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
        # 'SMPL-male_001', 'root', 
        'pelvis', 'left_hip', 'left_knee', 
        'left_ankle', 'left_foot', 'right_hip', 
        'right_knee', 'right_ankle', 'right_foot', 
        'spine1', 'spine2', 'spine3', 
        'neck', 'jaw', 
        'left_collar', 'left_shoulder', 'left_elbow', 
        'left_wrist', 'left_thumb', 'right_collar', 
        'right_shoulder', 'right_elbow', 'right_wrist', 'right_thumb')

disastejoint2smpljoint3D = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
disastejoint2smpljoint2D = [15, 7, 13, 18, 10, 22, 1, 4, 16, 20, 12, 0, 8, 14, 19, 11, 23, 2, 5, 17, 21, 3, 6, 9, 24]

def YawPitchRoll_to_RotMat(theta_yprs):
    '''
    calculate rotation matrix from order roll -> yaw -> pitch
    '''
    rot_mats = []
    for theta_ypr in theta_yprs:
        rot_mat = []
        for ypr in theta_ypr:
            roll = math.radians(ypr[2])
            yaw = math.radians(ypr[0])
            pitch = math.radians(ypr[1])
            R_roll = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
            R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            R_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
            R = R_yaw @ R_pitch @ R_roll
            rot_mat.append(R)
        rot_mats.append(rot_mat)
    return rot_mats

class DisasterDatasetLoader():
    def __init__(self, rootpath):
        #file path
        self.rootpath = rootpath
        self.sequences = self.get_all_sequences()
        self.sequences = sorted(self.sequences)
        self.img_names, self.img_paths = self.get_all_images()
        self.cam_filepaths = self.get_cam_filepaths()
        self.uv_joint_filepaths = self.get_uv_joint_filepaths()
        self.mask_paths = self.get_all_mask()
        self.img_paths = self.order_filepath(self.img_paths)
        self.cam_filepaths = self.order_filepath(self.cam_filepaths)
        self.uv_joint_filepaths = self.order_filepath(self.uv_joint_filepaths)
        self.mask_paths = self.order_filepath(self.mask_paths)
        self.img_names = self.order_filepath(self.img_names)
        # index
        # self.image_ids = self.create_id(lenth = len(self.img_names))
        # self.anno_ids = self.create_id(lenth = len(self.img_names), images=False)
        # cam intrensic
        self.hw_princpts = self.get_height_width()  #pixel
        self.focal_length = self.get_focal_length()  #19.074556 mm   about 1000 pixel
        #cam extrensic
        self.cam_locs, self.cam_yprs = self.get_cam_pose()
        self.cam_rots = YawPitchRoll_to_RotMat(self.cam_yprs)
        #2d joint
        self.joint_visibles, self.uv_joints = self.get_uv_joints()
        self.uv_joints, self.joint_visibles = self.orderUVjoint2SMPL()
        #smpl
        self.betas = self.get_beta()
        self.joint_locs_world, self.thetas_ypr ,self.genders_trans = self.get_theta_joings()
        self.joint_locs_world, self.thetas_ypr = self.order3Djoint2SMPL()
        self.rot_mats = YawPitchRoll_to_RotMat(self.thetas_ypr)
        #bind box
        self.bboxs = self.get_binding_box()
        #create folder (only need to exec once)
        # self.makedir4mask()
        # self.makedir4mask_2()

    def get_all_sequences(self):
        '''
        return subfolder names in dataset
        '''
        subfolders = [ name for name in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, name))]
        return subfolders

    def get_all_images(self):
        '''
        return all iamge names in folder_path
        '''
        names = []
        img_paths = []
        for s in self.sequences:
            img_path = []
            name = []
            path = os.path.join(rootpath, s, 'images')
            # root path, sub-file-folder, file-list
            for root, dirs, file_names in os.walk(path):
                for file_name in file_names:
                    name.append(file_name)
                    img_path.append(os.path.join(rootpath, s, 'images', file_name))
            names.append(name)
            img_paths.append(img_path)
        return names, img_paths

    def get_cam_filepaths(self):
        '''
        return all cam filepath
        '''
        cam_filepaths = []
        for s in self.sequences:
            cam_filepath = []
            path = os.path.join(rootpath, s, 'camera')
            for root, dirs, file_names in os.walk(path):
                for file_name in file_names:
                    cam_filepath.append(os.path.join(path, file_name))
            cam_filepaths.append(cam_filepath)
        return cam_filepaths
    
    def get_uv_joint_filepaths(self):
        '''
        return all 2d joint filepath
        '''
        uv_joint_filepaths =[]
        for s in self.sequences:
            uv_joint_filepath = []
            path = os.path.join(rootpath, s, '2Djoint')
            for root, dirs, file_names in os.walk(path):
                for file_name in file_names:
                    uv_joint_filepath.append(os.path.join(path, file_name))
            uv_joint_filepaths.append(uv_joint_filepath)
        return uv_joint_filepaths

    def get_all_mask(self):
        '''
        reeturn path and name of all mask images
        '''
        mask_paths = []
        mask_names = []
        for s in self.sequences:
            mask_path = []
            mask_name = []
            path = os.path.join(rootpath, s, 'mask')
            for root, dirs, file_names in os.walk(path):
                for file_name in file_names:
                    mask_path.append(os.path.join(rootpath, s, 'mask', file_name))
                    mask_name.append(file_name)
            mask_paths.append(mask_path)
            mask_names.append(mask_name)
        return mask_paths

    def order_filepath(self, paths):
        '''
        oder the file path first by length then by number
        '''
        sorted_lists = []
        for path in paths:
            sorted_list = sorted(path, key=lambda x: (len(x), x))
            sorted_lists.append(sorted_list)
        return sorted_lists
    
    def create_id(self, lenth, images=True):
        '''
        return a list of id with ginen lenth
        '''
        if images:
            ids = [i for i in range(IMAGE_IDS, 10000+lenth)] 
        else:
            ids = [i for i in range(ANNOTATION_IDS, 20000+lenth)]
        return ids

    def get_height_width(self):
        '''
        return height and width of the image in the list under given folder
        '''
        hw_princpts = []
        for img_path in self.img_paths:
            hw_princpt = []
            for path in img_path:
                with Image.open(path) as img:
                    w,h = img.size
                    princpt = [h/2, w/2]
                    hw_princpt.append([h, w, princpt])
            hw_princpts.append(hw_princpt)
        return hw_princpts

    def get_focal_length(self):
        '''
        calculate focal length from sensor width and FOV
        '''
        for s in self.sequences:
            path = os.path.join(rootpath, s, 'cam_info.txt')
            with open(path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            FOV = float(lines[0])
            SENSOR_WIDTH = float(lines[1])
            radian = math.radians(FOV)
            tan = math.tan(radian / 2)
            f = SENSOR_WIDTH / (2 * tan)
            f = f / Scale_Factor
        return f

    def get_cam_pose(self):
        '''
        return all camera translation and roll-yaw-pitch
        '''
        cam_locs = []
        cam_rots = []
        for cam_filepath in self.cam_filepaths:
            cam_loc = []
            cam_rot = []
            for path in cam_filepath:
                df = pd.read_csv(path)
                for index, row in df.iterrows():
                    cam_loc.append([row['LocationX'], row['LocationY'], row['LocationZ']])
                    cam_rot.append([row['RotationYaw'], row['RotationPitch'], row['RotationRoll']])
            cam_locs.append(cam_loc)
            cam_rots.append(cam_rot)
        return cam_locs, cam_rots

    def get_uv_joints(self):
        '''
        return 2d joint position and visibility
        '''
        joint_visibles = []
        uv_joints = []
        for uv_joint_filepath in self.uv_joint_filepaths:
            joint_visible_in_all_file = []
            uv_joint_in_all_file = []
            for path in uv_joint_filepath:
                joint_visible_in_1_file = []
                uv_joint_in_1_file = []
                df = pd.read_csv(path, header = None, names=['Name', 'Visible', 'U', 'V'])
                for index, row in df.iterrows():
                    joint_visible_in_1_file.append([row['Name'], row['Visible']])
                    uv_joint_in_1_file.append([row['U'], row['V']])
                joint_visible_in_all_file.append(joint_visible_in_1_file)
                uv_joint_in_all_file.append(uv_joint_in_1_file)
            joint_visibles.append(joint_visible_in_all_file)
            uv_joints.append(uv_joint_in_all_file)
        return joint_visibles, uv_joints

    def get_beta(self):
        '''
        return beta from all sequences
        '''
        betas = []
        for s in self.sequences:
            beta = []
            df = pd.read_csv(os.path.join(rootpath, s, s+'-beta.csv'), header = None, names=['value'])
            for value in df['value']:
                beta.append(value)
            betas.append(beta)
        return betas

    def get_theta_joings(self): 
        '''
        return theta and joint loation from all sequence
        '''
        genders_trans = []
        thetas = []
        joint_locs = []
        for s in self.sequences:
            theta = []
            joint_loc = []
            gender_tran = []
            df = pd.read_csv(os.path.join(rootpath, s, s+'-joint.csv'))
            for index, row in df.iterrows():
                if index == 0 or index == 1: gender_tran.append([row['Name'],row['LocationX'], row['LocationY'], row['LocationZ']])
                else :
                    joint_loc.append([row['LocationX'], row['LocationY'], row['LocationZ']])
                    theta.append([row['RotationYaw'], row['RotationPitch'], row['RotationRoll']])
            joint_locs.append(joint_loc)
            thetas.append(theta)
            genders_trans.append(gender_tran)
        return joint_locs, thetas, genders_trans

    def get_binding_box(self):
        '''
        num_conn: number of connected component
        labels_stats: array with same size of image, from 0
        stats: xywhs for every connected component, last item is whole image. s is number of pixel of each connected component
        centroids: center of each connected component

        return binding box for all images in x,y,w,h
        '''
        bbs = []
        for mask_path in self.mask_paths:
            bb = []
            for path in mask_path:
                mask = cv2.imread(path)
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY_INV) 
                num_conn, labels_stats, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity = 8)
                x, y, w, h = stats[0][0], stats[0][1], stats[0][2], stats[0][3]
                bb.append([x, y, w, h])

            bbs.append(bb)
        return bbs

    def makedir4mask(self):
        '''
        make filefolder for saving 
        '''
        for s in self.sequences:
            pth = os.path.join('./DisasterSet', 'bindingbox', s)
            os.makedirs(pth, exist_ok=True)
            print(f"文件夹已创建：{pth}")
        return

    def makedir4mask_2(self):
        '''
        make filefolder for saving bb on image and mask
        '''
        for s in self.sequences:
            pth1 = os.path.join('./DisasterSet', 'bindingbox', s, 'images')
            pth2 = os.path.join('./DisasterSet', 'bindingbox', s, 'masks')
            os.makedirs(pth1, exist_ok=True)
            os.makedirs(pth2, exist_ok=True)
            print(f"文件夹已创建：{pth1}")
            print(f"文件夹已创建：{pth2}")
        return

    def order3Djoint2SMPL(self):
        '''
        arrange the 3d joint order into SMPL
        '''
        results1, results2 = [], []
        for joint_loc_world in self.joint_locs_world:
            result1 = [None] * len(disastejoint2smpljoint3D)
            for pos, value in zip(disastejoint2smpljoint3D, joint_loc_world):
                result1[pos] = value
            results1.append(result1)
        for theta_ryp in self.thetas_ypr:
            result2 = [None] * len(disastejoint2smpljoint3D)
            for pos, value in zip(disastejoint2smpljoint3D, theta_ryp):
                result2[pos] = value
            results2.append(result2)
        return results1, results2

    def orderUVjoint2SMPL(self):
        '''
        arrange the uv joint order into SMPL
        '''
        results1, results2 = [], []
        for sequence_ in self.uv_joints:
            for image_ in sequence_:
                result1 = [None] * len(disastejoint2smpljoint2D)
                for pos, value in zip(disastejoint2smpljoint2D, image_):
                    result1[pos] = value
                results1.append(result1)
        for sequence_ in self.joint_visibles:
            for image_ in sequence_:
                result2 = [None] * len(disastejoint2smpljoint2D)
                for pos, value in zip(disastejoint2smpljoint2D, image_):
                    result2[pos] = value
                results2.append(result2)
        return results1, results2


import json
import os
from PIL import Image
import math
import pandas as pd
import numpy as np
import cv2
import pickle
from smpl_np import SMPLModel
from DisasterLoader import DisasterDatasetLoader
from scipy.spatial.transform import Rotation as R

FOV = 65.579994  #degree
SENSOR_WIDTH = 24.576 #mm
Folcal_Length = 19.074556 #mm
Scale_Factor = 0.0192

rootpath = "./DisasterSet/dataset"
IMAGE_IDS = 10000
ANNOTATION_IDS = 20000



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

def cal_uvd_joint(joint_cam, joint_uv):
    uvd_joints = []
    for joint, uv in zip(joint_cam, joint_uv):
        depth = (joint[2] - Folcal_Length)
        uvd_joints.append([uv[0], uv[1], depth])  # pixel, pixel, mm
    return uvd_joints

def make_json(img_paths, 
              img_names, 
              sequences, 
              hw_princpts, 
              focal_length, 
              joint_locs_world, 
              cam_locs, 
              cam_rots, 
              uv_joints, 
              genders_trans,
              betas,
              rot_mats):
    '''all image path and name and sequences'''
    image_path = []
    image_name = []
    sequence = []
    focal = []
    for s in img_paths:
        s_index = 0
        for i in range(len(s)): 
            sequence.append(sequences[s_index])
            focal.append([focal_length, focal_length])
        s_index += 1
        for pth in s:
            image_path.append(pth)
    for s in img_names:
        for name in s:
            image_name.append(pth)
    '''index for all images and annotations'''
    image_id = np.arange(10000, 10000+len(image_name))
    ann_id = np.arange(30000, 30000+len(image_name))
    '''width and height and princpt for all images'''
    width = []
    height = []
    princpt = []
    for s in hw_princpts:
        for hw_p in s:
            width.append(hw_p[0])
            height.append(hw_p[1])
            princpt.append(hw_p[2])
    '''joint location in cam'''
    smpl_joint_cams = []
    cam_loc = []
    cam_rot = []
    for s1, s2 in zip(cam_locs, cam_rots):
        for locs in s1: cam_loc.append(locs)
        for rots in s2: cam_rot.append(rots)
    for sequence_ in joint_locs_world:
        for loc,rot in zip(cam_loc, cam_rot):
            joints_cam = cal_joint_cam(loc, rot, sequence_)
            smpl_joint_cams.append(joints_cam)
    '''joint location in uvd format'''
    smpl_joints_imgs = []
    for joint_cam, joint_uv in zip(smpl_joint_cams, uv_joints):
        smpl_joints_img = cal_uvd_joint(joint_cam, joint_uv)
        smpl_joints_imgs.append(smpl_joints_img)
    '''gender and translation'''
    sequence_elements = {element: sequence.count(element) for element in sequence}
    sequence_elements_l = [value for value in sequence_elements.values()]
    genders = []
    trans = []
    for sequence_ in genders_trans:
        s_index = 0
        gender_tran = sequence_[0]
        gender = gender_tran[0]
        gender = gender[5 : -4]
        tran = [gender_tran[1], gender_tran[2], gender_tran[3]]
        for i in range(sequence_elements_l[s_index]):
            genders.append(gender)
            trans.append(tran)
        s_index += 1
    '''shape parameter'''
    beta = []
    for sequence_ in betas:
        s_index = 0
        for i in range(sequence_elements_l[s_index]): beta.append(sequence_)
        s_index += 1
    '''pose parameter'''
    theta = []
    for sequence_ in rot_mats:
        s_index = 0
        for i in range(sequence_elements_l[s_index]): theta.append(sequence_)
        s_index += 1
    return image_path, image_name, sequence, focal, image_id, ann_id, width, height, princpt, smpl_joint_cams, smpl_joints_imgs, genders, trans, beta, theta




diasdterset = DisasterDatasetLoader(rootpath=rootpath)
#file path
sequences = diasdterset.sequences
img_names = diasdterset.img_names
img_paths = diasdterset.img_paths
cam_filepaths = diasdterset.cam_filepaths
uv_joint_filepaths = diasdterset.uv_joint_filepaths
mask_paths = diasdterset.mask_paths
# index(error)
# image_ids = diasdterset.image_ids
# anno_ids = diasdterset.anno_ids
# cam intrensic
hw_princpts = diasdterset.hw_princpts  #pixel
focal_length = diasdterset.focal_length  #19.074556 mm   about 1000 pixel
#cam extrensic
cam_locs = diasdterset.cam_locs   #cam xyz     #sequences-files-camera-metrix
cam_yprs = diasdterset.cam_yprs   #cam angle   
cam_rots = diasdterset.cam_rots   #cam rot mat   #sequences-files-camera-metrix
#2d joint
joint_visibles = diasdterset.joint_visibles
uv_joints = diasdterset.uv_joints   #pixel
#smpl
betas = diasdterset.betas
thetas_ypr = diasdterset.thetas_ypr
joint_locs_world = diasdterset.joint_locs_world   #sequence-file-joint
rot_mats = diasdterset.rot_mats
genders_trans = diasdterset.genders_trans
#bind box
bboxs = diasdterset.bboxs
#same binding box (only need to exec once)
# save_img_with_bb(sequences, bbs, img_paths, mask_paths)


image_path, image_name, sequence, focal, image_id, ann_id, width, height, princpt, smpl_joint_cams, smpl_joints_imgs, genders, trans, beta, theta = make_json(img_paths, 
          img_names, 
          sequences, 
          hw_princpts, 
          focal_length, 
          joint_locs_world, 
          cam_locs, 
          cam_rots, 
          uv_joints, 
          genders_trans,
          betas,
          rot_mats)

# print(smpl_joint_cams[0])
# print(smpl_joints_imgs[0])




file_name = 'image_00000.jpg'   #2 people
id = 26413
width = 1920
height = 1080
focal = [1969,1961]
princpt = [960.0, 540.0]  #center pixel


ann_id = 32664
image_id = 26413
smpl_pose = [-2.862508773803711 , -0.07329627126455307 , 0.9958909749984741 ,
            -0.07329627126455307 , 0.9958909749984741 , 0.0745115876197815 ,
            0.9958909749984741 , 0.0745115876197815 , 0.07909861207008362 ,
            0.0745115876197815 , 0.07909861207008362 , -0.06977559626102448 ,
            0.07909861207008362 , -0.06977559626102448 , 0.032162416726350784 ,
            -0.06977559626102448 , 0.032162416726350784 , -0.05993887037038803 ,
            0.032162416726350784 , -0.05993887037038803 , 0.07062649726867676 ,
            -0.05993887037038803 , 0.07062649726867676 , -0.09686556458473206 ,
            0.07062649726867676 , -0.09686556458473206 , -0.0022304963786154985 ,
            -0.09686556458473206 , -0.0022304963786154985 , 0.009230132214725018 ,
            -0.0022304963786154985 , 0.009230132214725018 , -0.05775108560919762 ,
            0.009230132214725018 , -0.05775108560919762 , 0.0065861898474395275 ,
            -0.05775108560919762 , 0.0065861898474395275 , 0.04583512619137764 ,
            0.0065861898474395275 , 0.04583512619137764 , -0.025665011256933212 ,
            0.04583512619137764 , -0.025665011256933212 , 0.0285485852509737 ,
            -0.025665011256933212 , 0.0285485852509737 , -0.05498158931732178 ,
            0.0285485852509737 , -0.05498158931732178 , -0.0463797003030777 ,
            -0.05498158931732178 , -0.0463797003030777 , -0.002098375465720892 ,
            -0.0463797003030777 , -0.002098375465720892 , 0.006603514309972525 ,
            -0.002098375465720892 , 0.006603514309972525 , -0.08685794472694397 ,
            0.006603514309972525 , -0.08685794472694397 , -0.0221639946103096 ,
            -0.08685794472694397 , -0.0221639946103096 , 0.0552547425031662 ,
            -0.0221639946103096 , 0.0552547425031662 , -0.04769919067621231 ,
            0.0552547425031662 , -0.04769919067621231 , -0.026159651577472687 ,
            -0.04769919067621231 , -0.026159651577472687 , -0.016286522150039673 ,
            -0.026159651577472687 , -0.016286522150039673 , -0.015338781289756298 ,
            -0.016286522150039673 , -0.015338781289756298 , -0.0014880025992169976 ,
            -0.015338781289756298 , -0.0014880025992169976 , -0.0008256247965618968 ,
            -0.0014880025992169976 , -0.0008256247965618968 , 0.00610690750181675 ,
            -0.0008256247965618968 , 0.00610690750181675 , 0.0038399251643568277 ,
            0.00610690750181675 , 0.0038399251643568277 , -0.03684380650520325 ,
            0.0038399251643568277 , -0.03684380650520325 , 0.001555163529701531 ,
            -0.03684380650520325 , 0.001555163529701531 , 0.02989363856613636 ,
            0.001555163529701531 , 0.02989363856613636 , -0.004275277256965637 ,
            0.02989363856613636 , -0.004275277256965637 , 0.02251742035150528 ,
            -0.004275277256965637 , 0.02251742035150528 , 0.06968417018651962 ,
            0.02251742035150528 , 0.06968417018651962 , -0.06809979677200317 ,
            0.06968417018651962 , -0.06809979677200317 , 0.08894485980272293 ,
            -0.06809979677200317 , 0.08894485980272293 , -0.15977133810520172 ,
            0.08894485980272293 , -0.15977133810520172 , -0.38182905316352844 ,
            -0.15977133810520172 , -0.38182905316352844 , 0.08557619154453278 ,
            -0.38182905316352844 , 0.08557619154453278 , 0.1428789347410202 ,
            0.08557619154453278 , 0.1428789347410202 , 0.35358095169067383 ,
            0.1428789347410202 , 0.35358095169067383 , 0.16572438180446625 ,
            0.35358095169067383 , 0.16572438180446625 , -0.003665004391223192 ,
            0.16572438180446625 , -0.003665004391223192 , 0.08322486281394958 ,
            -0.003665004391223192 , 0.08322486281394958 , 0.17551420629024506 ,
            0.08322486281394958 , 0.17551420629024506 , -0.046808551996946335 ,
            0.17551420629024506 , -0.046808551996946335 , -1.1919732093811035 ,
            -0.046808551996946335 , -1.1919732093811035 , 0.11371263116598129 ,
            -1.1919732093811035 , 0.11371263116598129 , 0.06886783987283707 ,
            0.11371263116598129 , 0.06886783987283707 , 1.1696789264678955 ,
            0.06886783987283707 , 1.1696789264678955 , -0.3522104322910309 ,
            1.1696789264678955 , -0.3522104322910309 , -0.15195874869823456 ,
            -0.3522104322910309 , -0.15195874869823456 , 0.008579540997743607 ,
            -0.15195874869823456 , 0.008579540997743607 , -0.3498617708683014 ,
            0.008579540997743607 , -0.3498617708683014 , 0.05639911815524101 ,
            -0.3498617708683014 , 0.05639911815524101 , 0.06443024426698685 ,
            0.05639911815524101 , 0.06443024426698685 , -0.030834736302495003 ,
            0.06443024426698685 , -0.030834736302495003 , 0.003546711290255189 ,
            -0.030834736302495003 , 0.003546711290255189 , -0.09891810268163681 ,
            0.003546711290255189 , -0.09891810268163681 , -0.12474067509174347 ,
            -0.09891810268163681 , -0.12474067509174347 , -0.007357906550168991 ,
            -0.12474067509174347 , -0.007357906550168991 , 0.02747093141078949 ,
            -0.007357906550168991 , 0.02747093141078949 , -0.17097416520118713 ,
            0.02747093141078949 , -0.17097416520118713 , -0.03485023230314255 ,
            -0.17097416520118713 , -0.03485023230314255 , -0.14716947078704834 ,
            -0.03485023230314255 , -0.14716947078704834 , -0.10719497501850128 ,
            -0.14716947078704834 , -0.10719497501850128 , 0.10249000787734985 ,
            -0.10719497501850128 , 0.10249000787734985 , 0.1852644979953766]  #72
smpl_beta = [0.9100562930107117, 0.4227640926837921, 1.39877188205719, -1.547849416732788, 0.31586116552352905, 
             0.9757591485977173, 0.7573679089546204, 0.012063476257026196, -1.3912534713745117, -0.7965681552886963]
smpl_trans = [1.1488215684773742, 0.4564225816253239, 5.651787943524597]
gender = 'male'
bbox = [1290.9549560546875, 359.2942810058594, 155.12698364257812, 631.8033447265625]   #xywh
fitted_3d_pose = [[1.1488154299556619, 0.22674089605842174, 5.68205287059891] ,
                    [1.2110617592813788, 0.33005259866748393, 5.651529903841493] ,
                    [1.0925605058553038, 0.3174293630841786, 5.741773060840843] ,
                    [1.166484779107791, 0.10804463858638347, 5.692788089943407] ,
                    [1.238022050249797, 0.7322307669642979, 5.727464608234642] ,
                    [1.096525846410018, 0.7268578552726322, 5.820982224625824] ,
                    [1.177539805311185, -0.030034706639904374, 5.68216539485502] ,
                    [1.2459271609666167, 1.1541080021384769, 5.838487363738296] ,
                    [1.1421527437748729, 1.1556371593001895, 5.895717657489059] ,
                    [1.1654035378130494, -0.08639518980946004, 5.6599077874248005] ,
                    [1.2001341529073535, 1.2187721275809817, 5.7203703740065075] ,
                    [1.0313492327811538, 1.2115142488006168, 5.823566980046508] ,
                    [1.2142937689902602, -0.30850623612369954, 5.700256447625635] ,
                    [1.2636640503885566, -0.2071928239342159, 5.639428747204421] ,
                    [1.1366451442124663, -0.2152618027206844, 5.745669066292522] ,
                    [1.19789627789281, -0.374806747483868, 5.652457054090974] ,
                    [1.3336772680165587, -0.19721315865482747, 5.57446342645335] ,
                    [1.0509508386137305, -0.22210352187122762, 5.803047812861679] ,
                    [1.3454624771954833, 0.07346965308223308, 5.619254137707231] ,
                    [1.0484868645551024, 0.04620216722522319, 5.834032572192428] ,
                    [1.3222061753156005, 0.3322249495509678, 5.603347829860923] ,
                    [1.0235478043439208, 0.3149859451774174, 5.839860043806312] ,
                    [1.3073119133593856, 0.41763877804313243, 5.62256911335635] ,
                    [1.031164257216674, 0.40408282662663997, 5.839078954738853]
                    ]
smpl_joint_cam = [1.1488155126571655, 0.22674089670181274, 5.6820526123046875, 
                  1.2110618352890015, 0.33005261421203613, 5.651529788970947, 
                  1.0925605297088623, 0.31742942333221436, 5.741772651672363, 
                  1.1664848327636719, 0.10804462432861328, 5.692788124084473, 
                  1.2380220890045166, 0.7322307825088501, 5.727464199066162, 
                  1.0965259075164795, 0.7268580198287964, 5.820981979370117, 
                  1.1775398254394531, -0.030034750699996948, 5.682165145874023, 
                  1.2459272146224976, 1.1541080474853516, 5.838487148284912, 
                  1.1421527862548828, 1.1556373834609985, 5.895717620849609, 
                  1.1654036045074463, -0.086395263671875, 5.659907817840576, 
                  1.2001341581344604, 1.2187721729278564, 5.720370292663574, 
                  1.0313493013381958, 1.2115144729614258, 5.823566913604736, 
                  1.2142938375473022, -0.3085063695907593, 5.70025634765625, 
                  1.2636640071868896, -0.20719295740127563, 5.639428615570068, 
                  1.1366451978683472, -0.21526187658309937, 5.745668888092041, 
                  1.1978963613510132, -0.3748069405555725, 5.652456760406494, 
                  1.3336772918701172, -0.1972132921218872, 5.574463367462158, 
                  1.0509508848190308, -0.22210359573364258, 5.8030476570129395, 
                  1.345462441444397, 0.0734696090221405, 5.619254112243652, 
                  1.0484869480133057, 0.04620218276977539, 5.8340325355529785, 
                  1.3222062587738037, 0.33222493529319763, 5.6033477783203125, 
                  1.023547887802124, 0.31498605012893677, 5.839859962463379, 
                  1.3073118925094604, 0.41763871908187866, 5.6225690841674805, 
                  1.031164288520813, 0.4040829539299011, 5.839078903198242]
smpl_joint_img = [1358.1453857421875, 618.2872314453125, 5.6820526123046875, 
                  1381.98486328125, 654.5733642578125, 5.651529788970947, 
                  1334.710693359375, 648.4595336914062, 5.741772651672363, 
                  1363.506591796875, 577.2344360351562, 5.692788124084473, 
                  1385.6597900390625, 790.8141479492188, 5.727464199066162, 
                  1330.953369140625, 784.973876953125, 5.820981979370117, 
                  1368.09228515625, 529.6300659179688, 5.682165145874023, 
                  1380.2318115234375, 927.8042602539062, 5.838487148284912, 
                  1341.4908447265625, 924.5487060546875, 5.895717620849609, 
                  1365.474609375, 510.0534362792969, 5.659907817840576, 
                  1373.144775390625, 957.989013671875, 5.720370292663574, 
                  1308.749267578125, 948.136962890625, 5.823566913604736, 
                  1379.4942626953125, 433.82159423828125, 5.70025634765625, 
                  1401.258544921875, 467.92138671875, 5.639428615570068, 
                  1349.56591796875, 466.4990539550781, 5.745668888092041, 
                  1377.3291015625, 409.912109375, 5.652456760406494, 
                  1431.1337890625, 470.5935974121094, 5.574463367462158, 
                  1316.634033203125, 464.91278076171875, 5.8030476570129395, 
                  1431.508544921875, 565.6504516601562, 5.619254112243652, 
                  1313.9083251953125, 555.5367431640625, 5.8340325355529785, 
                  1424.673828125, 656.319091796875, 5.6033477783203125, 
                  1305.1455078125, 645.8169555664062, 5.839859962463379, 
                  1417.8687744140625, 685.7244262695312, 5.6225690841674805, 
                  1307.7603759765625, 675.7664794921875, 5.839078903198242]
h36m_joints = [0.010800976306200027, -0.25635167956352234, 0.03972969949245453, 
                0.1422768235206604, -0.23409663140773773, -0.05164412036538124, 
                0.10050088912248611, 0.22850359976291656, 0.06090044602751732, 
                0.11980737000703812, 0.6565930843353271, 0.19932761788368225, 
                -0.12222364544868469, -0.2767011225223541, 0.13305418193340302, 
                -0.05381256714463234, 0.19548824429512024, 0.15692837536334991, 
                -0.005093351937830448, 0.6380387544631958, 0.2747925817966461, 
                0.04108373075723648, -0.5097438097000122, 0.0456891804933548, 
                0.05682582035660744, -0.7589815855026245, 0.034238703548908234, 
                0.03407677635550499, -0.8505221605300903, -0.027048151940107346, 
                0.05959422141313553, -0.9585503935813904, 0.0040295664221048355, 
                0.17439031600952148, -0.6875147223472595, -0.050277478992938995, 
                0.2164473533630371, -0.3974912762641907, -0.08097337186336517, 
                0.1758730560541153, -0.13879019021987915, -0.06480380892753601, 
                -0.06053142994642258, -0.7092441320419312, 0.1358664631843567, 
                -0.1355573683977127, -0.4275045692920685, 0.19432586431503296, 
                -0.15193921327590942, -0.1636309176683426, 0.16783320903778076]




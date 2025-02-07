from propose.datasets.disaster import *
from propose.datasets.pw3d import*
import json


# test = Disaster3D(cfg = './configs/smpl_hm_xyz.yaml',
#                   ann_file = 'trainset_1000file.json')

# test_3dpw = PW3D(cfg = './configs/smpl_hm_xyz.yaml',
#                   ann_file = 'trainset_1000file.json')



# test = Disaster3D(cfg = './configs/smpl_hm_xyz.yaml',
#                   ann_file = 'dataset_1file.json')

# test = Disaster3D(cfg = './configs/smpl_hm_xyz.yaml',
#                   ann_file = 'dataset_4file.json')

pth = 'trainset_1000file.json'
with open(pth, 'r', encoding='utf-8') as file:
    data = json.load(file)
anns = data['annotations']
print(len(anns))
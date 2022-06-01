from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from io import BytesIO
from ipdb import set_trace as st
import torch
import cv2

class MVSDataset(Dataset):
    def __init__(self, metapath, datapath, listfile=None, mode="test", nviews=3, ndepths=384, interval_scale=0.01, depth_min=1.5, **kwargs):
        super(MVSDataset, self).__init__()
        self.metapath = metapath
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.depth_min = depth_min

        assert self.mode == "test"
        # self.metas = self.build_list()
       

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            
            pair_file = "pair_{}.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.metapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
               
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)


    def read_pose_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        poses = np.fromstring(' '.join(lines[0:4]), dtype=np.float32, sep=' ').reshape((4, 4))
        extrinsics = np.linalg.inv(poses)
        intrinsics = np.array([875.000000, 0., 400.000000, 
                                    0., 875.000000, 400.000000,
                                    0., 0., 1.], dtype=np.float64).reshape((3,3))
        '''intrinsics[:2, :]/=4'''
        # assume no downsampling for now
        return intrinsics, extrinsics, poses
        
    def read_bottle_img(self, filename, half_res):
        img_png = Image.open(filename) # bottles: png
        # scale 0~255 to 0~1
        assert NotImplementedError("use all RGBA channels")
        st() # see whether this is 4 channels
        img_jpg = img_png.convert('RGB')
        np_img = np.array(img_jpg, dtype=np.float32) / 255.
        
        return np_img
    
    def read_bottle_img_RGBA(self, filename, half_res):
        img_png = Image.open(filename) # bottles: png
        # scale 0~255 to 0~1
        np_img = np.array(img_png, dtype=np.float32) / 255.
        if half_res:
            st() # no half res for bottles now
        
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)


    def load_bottle_data(self, args, half_res=False):
        datapath = args.datadir
        imgs = []
        intrinsics_list = []
        extrinsics_list = []
        poses_list = []

        counts = [0]
        print('Loading poses and rgb ...')
        for scan in ['0_train', '1_val']:
            for vid in range(3): # view_ids -> all views
                img_filename = os.path.join(datapath, 'rgb/{}_{:0>4}.png'.format(scan,vid))
                pose_filename = os.path.join(datapath, 'pose/{}_{:0>4}.txt'.format(scan,vid))

                imgs.append(self.read_bottle_img_RGBA(img_filename, half_res=half_res))
                intrinsics, extrinsics, poses = self.read_pose_file(pose_filename)

                intrinsics_list.append(intrinsics)
                extrinsics_list.append(extrinsics)
                poses_list.append(poses)
                # print(poses)
            counts.append(len(poses_list))
        # st()
        for scan in ['2_test']:
            for vid in args.test_vids:
                pose_filename = os.path.join(datapath, 'pose/{}_{:0>4}.txt'.format(scan,vid))
                intrinsics, extrinsics, poses = self.read_pose_file(pose_filename)
                intrinsics_list.append(intrinsics)
                extrinsics_list.append(extrinsics)
                poses_list.append(poses)
            counts.append(len(poses_list))

        print('Loading poses and rgb finished!')

        imgs = np.stack(imgs)
        intrinsics =np.stack(intrinsics_list) 
        extrinsics =np.stack(extrinsics_list)
        poses =np.stack(poses_list)
        
       
        # before downsampling processing
        _, h, w, _ = imgs.shape # b h w 3
        focal = intrinsics[0,0,0] # 875 should be 
         
        # downsampling to speed up debugging
        assert args.downsample_ratio >=1
        if args.downsample_ratio >1:
            factor = args.downsample_ratio
            print("Down sampling ratio is ", factor)
            h = h//factor
            w = w//factor
            focal = focal/factor
            imgs_down_res = np.zeros((imgs.shape[0], h, w, 4))
            for i, img in enumerate(imgs):
                imgs_down_res[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            imgs = imgs_down_res

        # after downsampling processing
        hwf = [int(h),int(w), focal] # [int, int, np.float64], a list

        # train/val/test
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

        # for video rendering
        render_poses = poses[:100,...]

        return {"imgs": imgs,
                "intrinsics":intrinsics, 
                "extrinsics":extrinsics, 
                "poses":poses,
                "render_poses":render_poses, 
                "hwf":hwf, 
                "i_split":i_split,
                }



if __name__ == "__main__":
    dataset = MVSDataset('metadata/bottles', "/path/to/your/dataset/bottles", 'metadata/bottles/test.txt', 'test', 3,
                         128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))

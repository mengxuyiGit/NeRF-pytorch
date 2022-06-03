from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from io import BytesIO
from ipdb import set_trace as st
import torch

class MVSDataset(Dataset):
    def __init__(self, metapath, datapath, listfile, mode, nviews, ndepths=384, interval_scale=0.01, depth_min=1.5, **kwargs):
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
        self.metas = self.build_list()
       

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

    def pose_spherical(self, theta, phi, radius):
        trans_t = lambda t : torch.Tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,t],
            [0,0,0,1]]).float()

        rot_phi = lambda phi : torch.Tensor([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1]]).float()

        rot_theta = lambda th : torch.Tensor([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1]]).float()     

        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w

    # def __getitem__(self, idx):
    #     '''
    #     meta: [(tuple)]
    #     meta = [('1_val',26, [25,24]),('1_val', 28, [29,30]), ('1_val', 73, [68,70])]
    #     '''
    #     # meta = self.metas[idx]
       
    #     # scan, ref_view, src_views = meta
       
    #     # use only the reference view and first nviews-1 source views
    #     # view_ids = [ref_view] + src_views[:self.nviews - 1]

    #     imgs = []
    #     mask = None
    #     depth = None
    #     depth_values = None
    #     proj_matrices = []
    #     intrinsics_list = []
    #     extrinsics_list = []
    #     poses_list = []

    #     for scan in ['0_train', '1_val']:
    #         for vid in range(3): # view_ids -> all views
    #             img_filename = os.path.join(self.datapath, 'rgb/{}_{:0>4}.png'.format(scan,vid))
    #             pose_filename = os.path.join(self.datapath, 'pose/{}_{:0>4}.txt'.format(scan,vid))
    #             # intrinsic_file =os.path.join(self.datapath, 'intrinsics.txt') 

    #             imgs.append(self.read_bottle_img(img_filename))
    #             intrinsics, extrinsics, poses = self.read_pose_file(pose_filename)

    #             intrinsics_list.append(intrinsics)
    #             extrinsics_list.append(extrinsics)
    #             poses_list.append(poses)
    #             print(intrinsics, extrinsics, poses)

    #             # multiply intrinsics and extrinsics to get projection matrix
    #             # proj_mat = extrinsics.copy()
    #             # proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
    #             # proj_matrices.append(proj_mat)


    #     imgs = np.stack(imgs)
    #     # proj_matrices = np.stack(proj_matrices)
    #     intrinsics =np.stack(intrinsics_list) 
    #     extrinsics =np.stack(extrinsics_list)
    #     poses =np.stack(poses_list)

    #     _, _, h, w = imgs.shape
    #     focal = intrinsics[0,0,0] # 875 should be 
    #     hwf = [int(h),int(w), focal] # [int, int, np.float64], a list

    #     # i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    #     # i_train, i_val, i_test = i_split
    #     # counts = [0, 100, 150, 200] # hard-coded to adapt bottles train dataset
    #     counts = [0, 3, 5, 6] # use samll 8 samples to test pipeline
    #     st()
    #     i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    #     # borrowed from the blender
    #     render_poses = torch.stack([self.pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    #     # TODO:
    #         # 1. add render_poses
    #         # 2. load all images, 
    #         #    and split into train/val/test
    #         # 3. hwf
    #     return {"imgs": imgs,
    #             # "proj_matrices": proj_matrices,
    #             "intrinsics":intrinsics, 
    #             "extrinsics":extrinsics, 
    #             "poses":poses,
    #             "render_poses":render_poses, 
    #             "hwf":hwf, 
    #             "i_split":i_split,
    #             }

    def load_bottle_data(self, args, datapath, half_res=False):
        
        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        intrinsics_list = []
        extrinsics_list = []
        poses_list = []

        # counts = [0, 180, 190, 200] # hard-coded to adapt bottles train dataset
        # counts = [0, 100, 190, 200] # hard-coded to adapt bottles train dataset
        counts = [0, 10, 19, 20]
        for scan in ['0_train', '1_val']:
            for vid in range(10): # view_ids -> all views
                img_filename = os.path.join(datapath, 'rgb/{}_{:0>4}.png'.format(scan,vid))
                pose_filename = os.path.join(datapath, 'pose/{}_{:0>4}.txt'.format(scan,vid))
                # intrinsic_file =os.path.join(self.datapath, 'intrinsics.txt') 

                # imgs.append(self.read_bottle_img(img_filename, half_res=half_res))
                imgs.append(self.read_bottle_img_RGBA(img_filename, half_res=half_res))
                intrinsics, extrinsics, poses = self.read_pose_file(pose_filename)

                intrinsics_list.append(intrinsics)
                extrinsics_list.append(extrinsics)
                poses_list.append(poses)
                # print(intrinsics, extrinsics, poses)
                print(poses)

                # multiply intrinsics and extrinsics to get projection matrix
                # proj_mat = extrinsics.copy()
                # proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
                # proj_matrices.append(proj_mat)


        imgs = np.stack(imgs)
        # proj_matrices = np.stack(proj_matrices)
        intrinsics =np.stack(intrinsics_list) 
        extrinsics =np.stack(extrinsics_list)
        poses =np.stack(poses_list)

        _, h, w, _ = imgs.shape # b h w 3
        focal = intrinsics[0,0,0] # 875 should be 
        hwf = [int(h),int(w), focal] # [int, int, np.float64], a list

        # i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        # i_train, i_val, i_test = i_split
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

        # borrowed from the blender
        # st()
        # render_poses = torch.stack([self.pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        # # st()
        # if args.render_inverse_y == True:
        #     render_poses[:, 0:3, 1:2]*=-1
        #     render_poses[:, 1:2, 3:]*=-1 # converted to openGL coord
        # if args.render_inverse_z == True:
        #     render_poses[:, 0:3, 2:3]*=-1
        #     render_poses[:, 2:3, 3:]*=-1 # converted to openGL coord
        # st()
        render_poses = poses[:100,...]
        

        # TODO:
            # 1. add render_poses
            # 2. load all images, 
            #    and split into train/val/test
            # 3. hwf
        return {"imgs": imgs,
                # "proj_matrices": proj_matrices,
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

import os
import smplx
import trimesh
import pickle
import torch
import json
import numpy as np
from tqdm import tqdm

def generate_smplx(smplx_model_path, param_root, output_data_root):

    human_list = sorted(os.listdir(param_root))
    n_humans = len(human_list)

    for human_idx, human in tqdm(enumerate(human_list), total=n_humans):

        print(human)
        # import pdb; pdb.set_trace()
        obj_path = param_root + '/{}/{}.obj'.format(human, human)
        obj_vertices = trimesh.load(obj_path).vertices
        
        smpl_obj_path = param_root + '/{}/{}_smplx.obj'.format(human, human)
        smpl_model_load = trimesh.load(smpl_obj_path)
        smpl_obj_vertices = smpl_model_load.vertices
                
        vy_max = np.max(obj_vertices[:, 1])
        vy_min = np.min(obj_vertices[:, 1])

        # import pdb; pdb.set_trace()
        human_height = 1.80 
        obj_vertices[:, :3] = obj_vertices[:, :3] / (vy_max - vy_min) * human_height

        smpl_obj_vertices[:, :3] = smpl_obj_vertices[:, :3] / (vy_max - vy_min) * human_height
        smpl_obj_vertices[:, 1] -= np.min(obj_vertices[:, 1])
        
        regenerated_smpl_mesh = trimesh.Trimesh(smpl_obj_vertices,
                                    smpl_model_load.faces,
                                    process=False,
                                    maintain_order=True)

        regenerated_mesh_fname = os.path.join(output_data_root, '{:04d}.obj'.format(human_idx))
        regenerated_smpl_mesh.export(regenerated_mesh_fname)

        print('human {}/{} processed!'.format(str(human_idx),n_humans))


if __name__ == '__main__':
    
    smplx_model_path = '/data/lujia/thuman2/THuman2.1/smplx_models'
    
    param_root = '/data/lujia/4D_dress'
    output_data_root = '/data/lujia/4D_dress_30_val/smplx_obj'
    if not os.path.exists(output_data_root):
        os.makedirs(output_data_root)

    generate_smplx(smplx_model_path, param_root, output_data_root)
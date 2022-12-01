


import jittor as jt

from jittor import nn
jt.flags.use_cuda = 1

import torch
import os
import tqdm
import numpy as np
import imageio
import cv2
import argparse
import jrender as jr


import time
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
from geomloss import SamplesLoss

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class SimpleModel(nn.Module):
    def __init__(self, template_path):
        super(SimpleModel, self).__init__()

        # set template mesh
        self.template_mesh = jr.Mesh.from_obj(template_path, dr_type='softras', load_texture=True, texture_res=5, texture_type='surface')
        self.vertices = self.template_mesh.vertices.stop_grad()
        self.faces = self.template_mesh.faces.stop_grad()
        self.textures = self.template_mesh.textures.stop_grad()

        # optimize offset
        self.displace = jt.zeros((1, 1, 1))+[0.0,0.0,0.4]

    def execute(self, batch_size):
        vertices = self.vertices+self.displace
        return jr.Mesh(vertices.repeat(batch_size, 1, 1), self.faces.repeat(batch_size, 1, 1), dr_type='softras', textures = self.textures)
def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = jt.cos(angle)
    sin = jt.sin(angle)
    one = jt.ones_like(angle)
    zero = jt.zeros_like(angle)
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")
    return jt.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, jt.unbind(euler_angles, -1))
    ]
    return jt.matmul(jt.matmul(matrices[0], matrices[1]), matrices[2])
class Furniture(nn.Module):
    def __init__(self, template_path):
        super(Furniture, self).__init__()
        meshes_path = [os.path.join(template_path,x) for x in os.listdir(template_path) if x.endswith(".obj")]
        self.meshes = []
        self.displaces = []
        self.rotations = []
        for meshpath in meshes_path:
            #print(meshpath)
            mesh = jr.Mesh.from_obj(meshpath, dr_type='softras', load_texture=True, texture_res=10, texture_type='surface')
            mesh.vertices = mesh.vertices.stop_grad()
            self.meshes.append(mesh)
            self.displaces.append(jt.zeros(3))
            self.rotations.append(jt.zeros(3))


    def transform(self, vertices, trans, rot):
        rot_center = jt.mean(vertices,dim=1,keepdims=True)
        vertices-=rot_center
        #rot = jt.array([0,3.1415926/2.0,0])
        #print(vertices)
        vertices = jt.matmul(vertices, euler_angles_to_matrix(rot).transpose(0,2,1))
        #print(vertices)
        vertices+=rot_center+trans
        return vertices
    
    def execute(self):
        meshlist = []
        for (mesh,trans,rot) in zip(self.meshes,self.displaces,self.rotations):
            trans[1] = 0
            rot[0] = 0.0
            rot[2]=0.0
            #print(trans,rot)
            new_vert = self.transform(mesh.vertices.clone(), trans, rot)
            adjust_mesh = jr.Mesh(new_vert, mesh.faces, dr_type='softras', textures = mesh.textures)
            meshlist.append(adjust_mesh)
            #print(new_vert)
        return jr.join_meshes_as_scene(meshlist,include_texture=True)


class Matcher():
    def __init__(self, res, device) -> None:
        self.loss = SamplesLoss("sinkhorn", blur=0.01)
        self.device=device
        self.resolution = res
        pass

    def match_Sinkhorn(self, haspos, render_point_5d, gt_rgb, view):
        h,w = render_point_5d.shape[1:3]
        target_point_5d = torch.zeros((haspos.shape[0], h, w, 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1)
        target_point_5d[..., 3:] = render_point_5d[...,3:].clone().detach()
        target_point_5d = target_point_5d.reshape(-1, h*w, 5)
        render_point_5d_match = render_point_5d.clone().reshape(-1,h*w,5)
        render_point_5d_match.clamp_(0.0,1.0)
        pointloss = self.loss(render_point_5d_match, target_point_5d)*self.resolution*self.resolution
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match])
        return (render_point_5d-g.reshape(-1,h,w,5)).detach()
    
    def match(self, gt_rgb, rgb, point, msk, view=0):
        render_point_5d = torch.cat([torch.clamp(rgb, 0.0,1.0)[...,:3], point], dim=-1)
        res = self.match_Sinkhorn(msk, render_point_5d, gt_rgb[...,:3], view)
        return res
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'source.npy'))
    parser.add_argument('-c', '--camera-input', type=str, 
        default=os.path.join(data_dir, 'camera.npy'))
    parser.add_argument('-t', '--template-mesh', type=str, 
        default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    parser.add_argument('-o', '--output-dir', type=str, 
        default=os.path.join(data_dir, 'results/output_drl_fur'))
    parser.add_argument('-b', '--batch-size', type=int,
        default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = Furniture(os.path.join(args.template_mesh,"gt"))
    # read training images and camera poses

    res=128
    sta = time.time()
    
    bg_img = cv2.imread(os.path.join(args.template_mesh,"init", "0_bg.png"))
    bg_img = cv2.resize(bg_img,(res,res))
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = jt.array(bg_img)/255.0

    cam = np.loadtxt(os.path.join(args.template_mesh,"gt","camera.txt")).tolist()
    pos = jt.array([cam[0][0],cam[0][2],-cam[0][1]])
    rot = jt.array([cam[1][0],cam[1][2],cam[1][1]])

    up = jt.array([0,-1,0.])
    rotm = euler_angles_to_matrix(rot,convention="XYZ")[0]
    dir = jt.matmul(up,rotm)
    dir[2]*=-1

    mesh = model()
    renderer = jr.Renderer(image_size=res, background_color=[1.0,1.0,1.0], sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='look', eye = pos, camera_direction = dir, viewing_angle=35, dr_type='DrL', coordinate = "left")
    images, points_orig, msks_orig = renderer.render_mesh(mesh, mode='rgb')
    images_gt = jt.array(images)
    images_gt = images_gt.permute(0,2,3,1)[...,:3]

    images_gt[msks_orig[:,0,...]==0] = bg_img[msks_orig[0,0]==0]
    image = images_gt[0].numpy()
    imageio.imsave(os.path.join(args.output_dir, 'gt.jpg'), (255*image[...,:3]).astype(np.uint8))

    #points_orig = (points_orig+1.0)/2.0
    #pos_vis = np.zeros((res,res,3))
    #pos_vis[...,:2] = points_orig.numpy().transpose(0, 2, 3, 1)[0]
    #pos_vis[...,:2] *= msks_orig.numpy().transpose(0, 2, 3, 1)[0]
    #print(image_gt_torch.shape)
    #cv2.imwrite("tmp.png",(pos_vis*255).astype(np.uint8))
    #exit()

    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'), mode='I')
    image_gt_torch = torch.from_numpy(images_gt.numpy())

    
    model = Furniture(os.path.join(args.template_mesh,"init")) 
    optimizer = nn.Adam(model.displaces+model.rotations, 0.02)
    matcher = Matcher(res,device)
    loop = tqdm.tqdm(list(range(0, 500)))
    
    for i in loop:
        mesh = model()
        images_pred, points, msks = renderer.render_mesh(mesh, mode='silhouettes')
        points = (points+1.0)/2.0
        images_pred = jt.array(images_pred).permute(0,2,3,1)[...,:3]
        points = jt.array(points).permute(0,2,3,1)
        msks = jt.array(msks).permute(0,2,3,1)
        images_pred[msks[...,0]==0] = bg_img[msks[0,...,0]==0]
        image_torch = torch.from_numpy(images_pred.data).to(device).requires_grad_()
        points_torch = torch.from_numpy(points.data).to(device).requires_grad_()
        msks_torch = torch.from_numpy(msks.data).to(device)
        
        match_res = jt.array(matcher.match(image_gt_torch, image_torch, points_torch, msks_torch).cpu().numpy())
        x = jt.concat([jt.array(images_pred[...,:3]),points], dim=-1)
        loss = jt.mean((x-match_res)**2)
            
        loop.set_description('Loss: %.4f'%(loss.item()))
        optimizer.step(loss)
        
        if i % 10 == 0:
            image = images_pred.numpy()[0][...,:3]
            imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.jpg'%i), (255*image).astype(np.uint8))
            writer.append_data((255*image).astype(np.uint8))
        
    writer.close()
    # save optimized mesh
    model().save_obj(os.path.join(args.output_dir, 'final.obj'), save_texture=True)
    print(f"Cost {time.time() - sta} secs.")


if __name__ == '__main__':
    main()
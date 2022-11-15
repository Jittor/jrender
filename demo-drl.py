
import torch
import jittor as jt
from jittor import nn
jt.flags.use_cuda = 1

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

class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

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
        default=os.path.join(data_dir, 'results/output_drl'))
    parser.add_argument('-b', '--batch-size', type=int,
        default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = Model(args.template_mesh)

    # read training images and camera poses

    print("start optim")
    res=512
    sta = time.time()
    
    mesh = model(args.batch_size)
    mesh.vertices[...,2]-=1.2
    num_views=args.batch_size
    elevations = jt.linspace(0, 360, num_views)
    azimuths = jt.linspace(-90, 180, num_views)
    camera_distances = jt.ones_like(azimuths)*5.0
    renderer = jr.Renderer(image_size=res, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15, dr_type='DrL')
    renderer.transform.set_eyes_from_angles(camera_distances, elevations, azimuths)
    images, points_orig, msks_orig = renderer.render_mesh(mesh, mode='rgb')
    
    points_orig = (points_orig+1.0)/2.0
    pos_vis = np.zeros((res,res,3))
    pos_vis[...,:2] = points_orig.numpy().transpose(0, 2, 3, 1)[0]
    pos_vis[...,:2] *= msks_orig.numpy().transpose(0, 2, 3, 1)[0]
    #cv2.imwrite("tmp.png",(pos_vis*255).astype(np.uint8))
    #exit()
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'), mode='I')
    images_gt = jt.array(images)
    image_gt_torch = torch.from_numpy(images_gt.numpy().transpose((0, 2, 3, 1)))
    optimizer = nn.Adam(model.parameters(), 0.01)
    matcher = Matcher(res,device)
    
    image = images_gt.permute(0,2,3,1).numpy()[0]
    imageio.imsave(os.path.join(args.output_dir, 'deform_gt.png'), (255*image).astype(np.uint8))
            

    loop = tqdm.tqdm(list(range(0, 1000)))
    for i in loop:
        mesh = model(args.batch_size)
        images_pred, points, msks = renderer.render_mesh(mesh, mode='silhouettes')
        points = (points+1.0)/2.0
        images_pred = jt.array(images_pred).permute(0,2,3,1)
        points = jt.array(points).permute(0,2,3,1)
        msks = jt.array(msks).permute(0,2,3,1)
        loss = jt.sum(images_pred)+jt.sum(points)*0
        optimizer.step(loss)
        continue

        image_torch = torch.from_numpy(images_pred.data).to(device).requires_grad_()
        points_torch = torch.from_numpy(points.data).to(device).requires_grad_()
        msks_torch = torch.from_numpy(msks.data).to(device)
        
        match_res = jt.array(matcher.match(image_gt_torch, image_torch, points_torch, msks_torch).cpu().numpy())
        x = jt.concat([jt.array(images_pred[...,:3]),points], dim=-1)
        loss = jt.mean((x-match_res)**2)
            
        loop.set_description('Loss: %.4f'%(loss.item()))
        optimizer.step(loss)
        
        #if i % 100 == 0:
        #    image = images_pred.numpy()[0]
        #    imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), (255*image).astype(np.uint8))
        #    writer.append_data((255*image).astype(np.uint8))
        #print(model.displace)
        
    writer.close()
    # save optimized mesh
    model(1).save_obj(os.path.join(args.output_dir, 'final.obj'), save_texture=True)
    print(f"Cost {time.time() - sta} secs.")


if __name__ == '__main__':
    main()
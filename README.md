# Jrender 2.0 (Jittor Rendering Library)

## News!

* SoftRas acceleration! (Ours is times faster than other implementations.)

* Support various rendering effects including **Ambient occlusion**, **Soft shadow**, **Global illumination** and **Subsurface scattering**.   

## Gallery

* **Ambient occlusion**

<p align="middle">
<img src="data/results/output_render/serapis_without_ssao.jpg" width="200" \>
<img src="data/results/output_render/serapis_with_ssao.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/buddha_without_ssao.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/buddha_with_ssao.jpg" width="200" style="padding-left: 5px;" \>
</p>
<pre>
     Without SSAO                 With SSAO                 Without SSAO                 With SSAO
</pre>

* **Soft Shadow**

<p align="middle">
<img src="data/results/output_render/cone_hard_shadow.jpg" width="200" \>
<img src="data/results/output_render/cone_soft_shadow.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/desk_hard_shadow.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/desk_soft_shadow.jpg" width="200" style="padding-left: 5px;" \>
</p>

<pre>
     Hard Shadow                 Soft Shadow                 Hard Shadow                 Soft Shadow
</pre>

* **Global Illumination**
<p align="middle">
<img src="data/results/output_render/cornellbox_with_ssr.jpg" width="200" \>
<img src="data/results/output_render/cornellbox_with_sssr.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/bunny_with_ssr.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/bunny_with_sssr.jpg" width="200" style="padding-left: 5px;" \>
</p>
 
<pre>
     Mirror Reflection          Glossy Reflection           Mirror Reflection         Glossy Reflection
</pre>

* **Subsurface Scattering**

<p align="middle">
<img src="data/results/output_render/noSSS.jpg" width="200" \>
<img src="data/results/output_render/withSSS.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/noSSS2.jpg" width="200" style="padding-left: 5px;" \>
<img src="data/results/output_render/withSSS2.jpg" width="200" style="padding-left: 5px;" \>
</p>

<pre>
     Without SSS                  With SSS                   Without SSS                  With SSS
</pre>

## Performance Comparisons

* SoftRas Rendering time for 1024 * 1024 resolution (including forward rendering + gradients: **ms**)

  |                   | Small-size Mesh(280) | Middle-size Mesh(3.3k) | Large-size Mesh(39k) |
  | ----------------- | :------------------: | :--------------------: | :------------------: |
  | Official SoftRas  |         9.2          |          44.1          |        404.9         |
  | Pytorch3D SoftRas |         48.3         |          53.3          |         82.9         |
  | Jrender SoftRas   |         7.3          |          11.5          |         35.5         |

* SoftRas **speedup** for 1024 * 1024 resolution compared with other implementations 

  |                   | Small-size Mesh(280) | Middle-size Mesh(3.3k) | Large-size Mesh(39k) |
  | ----------------- | :------------------: | :--------------------: | :------------------: |
  | Official SoftRas  |       **1.3**        |        **3.8**         |       **11.4**       |
  | Pytorch3D SoftRas |       **6.6**        |        **4.6**         |       **2.3**        |

* NMR Performance for 1024 * 1024 resolution (including forward rendering + gradients: **ms**)

  |              | Small-size Mesh(280) | Middle-size Mesh(3.3k) | Large-size Mesh(39k) |
  | ------------ | :------------------: | :--------------------: | :------------------: |
  | Official NMR |         42.3         |         236.9          |         2581         |
  | Jrender NMR  |         32.1         |          95.7          |        114.7         |
  | **Speedup**  |       **1.3**        |        **2.5**         |       **22.5**       |

## Introduction

Main features:

* Mesh loader and materials loader for OBJ file format；
* Surface Rendering & Volume Rendering；
* Optimized differentiable rendering algorithms including NMR and SoftRas；
* PBR materials & Ambient occlusions & Soft shadows & Global illuminations & Subsurface scatterings；
* Various loss functions and projection functions.

## Examples

## Basic Tutorials

- [Jrender 2.0 (Jittor rendering libary)](#jrender-20-jittor渲染库)
  - Basic Tutorials
    - Basic Tutorial 1：Rendering objects
    - Basic Tutorial 2：Geometry Optimization
    - Basic Tutorial 3：Rendering Specular materials
    - Basic Tutorial 4：Texture Optimization
    - Basic Tutorial 5：Metallic Texture Optimization
    - Basic Tutorial 6：Roughness Texture Optimization

## Advanced Tutorials

* Advanced Tutorial 1：3D Reconstruction
* Advanced Tutorial 2：NeRF

## Usage

Please install Jittor before using Jrender. Jittor could be installed from [this](https://github.com/Jittor/jittor)

And other dependent packages：

```
jittor
imageio==2.9.0
imageio-ffmpeg==0.4.3
matplotlib==3.3.0
configargparse==1.3
tensorboard==1.14.0
tqdm==4.46.0
opencv-python==4.2.0.34
```

After the installing, the following commands could be used to run these demos:

```
git clone https://github.com/jittor/jrender.git
cd jrender
python demo1-render.py
python demo2-deform
python demo3-render_specular.py
python demo4-optim_textures.py
python demo5-optim_metallic_textures.py
python demo6-optim_roughness_textures.py
```

## Basic Tutorials

### Basic Tutorial 1：Rendering objects

This tutorial is used to render a cow with texture based on Jrender. 

    import jrender as jr
    
    # create a mesh object from args.filename_input
    mesh = jr.Mesh.from_obj(args.filename_input, load_texture=True, texture_res=5, texture_type='surface', dr_type='softras')
    
    # create a softras using default parameters
    renderer = jr.Renderer(dr_type='softras')
    
    # set the position of eyes
    renderer.transform.set_eyes_from_angles(2.732, 30, 0)
    
    # render the given mesh to a rgb or silhouette image
    rgb = renderer.render_mesh(mesh)
    silhouettes = renderer.render_mesh(mesh, mode='silhouettes') # or mode = 'rgb'

The rendering results with texture and silhouettes，please refer the  [Code](https://github.com/Jittor/jrender/blob/main/demo1-render.py).

<p align="middle">
<img src="data/imgs/softras-rgb.gif" width="200" \>
<img src="data/imgs/softras-silhouettes.gif" width="200" style="padding-left: 25px;" \>
</p>

### Basic Tutorial 2：Geometry Optimization

This tutorial use the differentiable renderer to deform sphere to airplane.

    import jrender as jr
    from jrender import neg_iou_loss, LaplacianLoss, FlattenLoss
    
    class Model(nn.Module):
        def __init__(self, template_path):
            super(Model, self).__init__()
    
            # set template mesh
            self.template_mesh = jr.Mesh.from_obj(template_path, dr_type='softras')
            self.vertices = (self.template_mesh.vertices * 0.5).stop_grad()
            self.faces = self.template_mesh.faces.stop_grad()
            self.textures = self.template_mesh.textures.stop_grad()
    
            # optimize for displacement map and center
            self.displace = jt.zeros(self.template_mesh.vertices.shape)
            self.center = jt.zeros((1, 1, 3))
    
            # define Laplacian and flatten geometry constraints
            self.laplacian_loss = LaplacianLoss(self.vertices[0], self.faces[0])
            self.flatten_loss = FlattenLoss(self.faces[0])
    
        def execute(self, batch_size):
            base = jt.log(self.vertices.abs() / (1 - self.vertices.abs()))
            centroid = jt.tanh(self.center)
            vertices = (base + self.displace).sigmoid() * nn.sign(self.vertices)
            vertices = nn.relu(vertices) * (1 - centroid) - nn.relu(-vertices) * (centroid + 1)
            vertices = vertices + centroid
    
            # apply Laplacian and flatten geometry constraints
            laplacian_loss = self.laplacian_loss(vertices).mean()
            flatten_loss = self.flatten_loss(vertices).mean()
            return jr.Mesh(vertices.repeat(batch_size, 1, 1), 
                        self.faces.repeat(batch_size, 1, 1), dr_type='softras'), laplacian_loss, flatten_loss
    
    # define a softras render
    renderer = jr.SoftRenderer(image_size=64, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15, dr_type='softras')
    
    for i in range(1000):
        # get the deformede mesh object, laplacian_loss, flatten_loss
        mesh, laplacian_loss, flatten_loss = model(args.batch_size)
    
        # render silhouettes image
        images_pred = renderer.render_mesh(mesh, mode='silhouettes')
    
        loss = neg_iou_loss(images_pred, images_gt[:, 3]) + \
                0.03 * laplacian_loss + \
                0.0003 * flatten_loss
        optimizer.step(loss)

The optimization process from sphere to airplane is shown as followed，please refer the [Code](https://github.com/Jittor/jrender/blob/main/demo2-deform.py).

<p align="middle">
<img src="data/imgs/n3mr-deform.gif" width="200" style="max-width:50%;">
</p>

### Basic Tutorial 3：Rendering Specular Materials

We implement the PBR shading models based on the microfacet theory in Jrender, which could be used to render specular/glossy materials. And the users could control the various highlights and other shading appearances by modifying roughness and metallic. 

    # load from Wavefront .obj file
    mesh = jr.Mesh.from_obj(args.filename_input, load_texture=True, texture_res=5 ,texture_type='surface', dr_type='softras')
    
    # create renderer with SoftRas
    renderer = jr.Renderer(dr_type='softras')
    
    #Roughness/Metallic setup 0.5 0.4
    metallic_textures = jt.zeros((1, mesh.faces.shape[1], 5 * 5, 1)).float32() + 0.5
    roughness_textures = jt.zeros((1, mesh.faces.shape[1], 5 * 5, 1)).float32() + 0.4
    
    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'), mode='I')
    imgs = []
    from PIL import Image
    for num, azimuth in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        loop.set_description('Drawing rotation')
        renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        rgb = renderer(mesh.vertices, mesh.faces, textures=mesh.textures, metallic_textures=metallic_textures, roughness_textures=roughness_textures)
        image = rgb.numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

The rendering results with specular materials are shown as followed, please refer to [Code](https://github.com/Jittor/jrender/blob/main/demo3-render_specular.py).

<p align="middle">
<img src="data/imgs/specular.gif" width="200" style="max-width:50%;">
</p>

### Basic Tutorial 4：Texture Optimization

    class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
    
        # set template mesh
        self.template_mesh = jr.Mesh.from_obj(filename_obj, dr_type='softras')
        self.vertices = (self.template_mesh.vertices * 0.6).stop_grad()
        self.faces = self.template_mesh.faces.stop_grad()
        # self.textures = self.template_mesh.textures
        texture_size = 4
        self.textures = jt.zeros((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3)).float32()
    
        # load reference image
        self.image_ref = jt.array(imread(filename_ref).astype('float32') / 255.).permute(2,0,1).unsqueeze(0).stop_grad()
    
        # setup renderer
        self.renderer = jr.Renderer(camera_mode='look_at', perspective=False, light_intensity_directionals=0.0, light_intensity_ambient=1.0, dr_type='softras')
    
    def execute(self):
        num = np.random.uniform(0, 360)
        self.renderer.transform.set_eyes_from_angles(2.732, 0, num)
        image = self.renderer(self.vertices, self.faces, jt.tanh(self.textures))
        loss = jt.sum((image - self.image_ref).sqr())
        return loss
    
    model = Model(args.filename_obj, args.filename_ref)
    
    optimizer = nn.Adam([model.textures], lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(300))
    for num in loop:
        loop.set_description('Optimizing')
        loss = model()
        optimizer.step(loss)

The left image is the target texture and the right image shows the optimization process, please refer to [Code](https://github.com/jittor/jrender/blob/main/demo4-optim_textures.py).

<p align="middle">
<img src="data/ref/ref_texture.png" width="200" style="max-width:50%;">
<img src="data/imgs/optim_textures.gif" width="200" style="max-width:50%;">
</p>


### Basic Tutorial 5：Metallic Texture Optimization

    class Model(nn.Module):
        def __init__(self, filename_obj, filename_ref):
            super(Model, self).__init__()
    
            # set template mesh
            texture_size = 4
            self.template_mesh = jr.Mesh.from_obj(filename_obj, texture_res=texture_size,load_texture=True, dr_type='softras')
            self.vertices = (self.template_mesh.vertices).stop_grad()
            self.faces = self.template_mesh.faces.stop_grad()
            self.textures = self.template_mesh.textures.stop_grad()
            self.metallic_textures = jt.zeros((1, self.faces.shape[1], texture_size * texture_size, 1)).float32()
            self.roughness_textures = jt.zeros((1, self.faces.shape[1], texture_size * texture_size, 1)).float32() + 0.5
            self.roughness_textures = self.roughness_textures.stop_grad()
            # load reference image
            self.image_ref = jt.array(imread(filename_ref).astype('float32') / 255.).permute(2,0,1).unsqueeze(0).stop_grad()
            # setup renderer
            self.renderer = jr.Renderer(dr_type='softras', light_intensity_directionals=1.0, light_intensity_ambient=0.0)
    
        def execute(self):
            self.renderer.transform.set_eyes_from_angles(2.732, 30, 140)
            image = self.renderer(self.vertices, self.faces, self.textures, metallic_textures=self.metallic_textures, roughness_textures=self.roughness_textures)
            loss = jt.sum((image - self.image_ref).sqr())
            return loss


    model = Model(args.filename_obj, args.filename_ref)
    
    optimizer = nn.Adam([model.metallic_textures], lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(20))
    for num in loop:
        loop.set_description('Optimizing')
        loss = model()
        optimizer.step(loss)

The left image is the initial state，the middle image is the target image and the right image shows the optimization process. Please refer to [Code](https://github.com/jittor/jrender/blob/main/demo5-optim_metallic_textures.py).

<p align="middle">
<img src="data/ref/init_metallic.png" width="200" style="max-width:50%;">
<img src="data/ref/ref_metallic.png" width="200" style="max-width:50%;">
<img src="data/imgs/metallic.gif" width="200" style="max-width:50%;">
</p>



### Basic Tutorial 6：Roughness Texture Optimization
    class Model(nn.Module):
        def __init__(self, filename_obj, filename_ref):
            super(Model, self).__init__()
    
            # set template mesh
            texture_size = 4
            self.template_mesh = jr.Mesh.from_obj(filename_obj, texture_res=texture_size,load_texture=True, dr_type='softras')
            self.vertices = (self.template_mesh.vertices).stop_grad()
            self.faces = self.template_mesh.faces.stop_grad()
            self.textures = self.template_mesh.textures.stop_grad()
            self.metallic_textures = jt.zeros((1, self.faces.shape[1], texture_size * texture_size, 1)).float32() + 0.4
            self.metallic_textures = self.metallic_textures.stop_grad()
            self.roughness_textures = jt.ones((1, self.faces.shape[1], texture_size * texture_size, 1)).float32()
            # load reference image
            self.image_ref = jt.array(imread(filename_ref).astype('float32') / 255.).permute(2,0,1).unsqueeze(0).stop_grad()
            # setup renderer
            self.renderer = jr.Renderer(dr_type='softras')
    
        def execute(self):
            self.renderer.transform.set_eyes_from_angles(2.732, 30, 140)
            image = self.renderer(self.vertices, self.faces, self.textures, metallic_textures=self.metallic_textures, roughness_textures=self.roughness_textures)
            loss = jt.sum((image - self.image_ref).sqr())
            return loss
    
    def main():
        model = Model(args.filename_obj, args.filename_ref)
    
        optimizer = nn.Adam([model.roughness_textures], lr=0.1, betas=(0.5,0.999))
        loop = tqdm.tqdm(range(15))
        for num in loop:
            loop.set_description('Optimizing')
            loss = model()
            optimizer.step(loss)


The left image is the initial state，the middle image is the target image and the right image shows the optimization process. Please refer to [Code](https://github.com/jittor/jrender/blob/main/demo6-optim_roughness_textures.py).

<p align="middle">
<img src="data/ref/init_roughness.png" width="200" style="max-width:50%;">
<img src="data/ref/ref_roughness.png" width="200" style="max-width:50%;">
<img src="data/imgs/roughness.gif" width="200" style="max-width:50%;">
</p>


## Advanced Tutorials

### Advanced Tutorial 1：3D Reconstruction

We reimplement Wu's CVPR 2020 Best Paper, which use the differentiable rendering technique for 3D reconstruction. And our training speed is 1.31 times than the official version. Please refer to [Code](https://github.com/Jittor/unsup3d-jittor) for details.

### Advanced Tutorials 2：NeRF

We reimplement NeRF published in ECCV 2020，which represents 3D scenes with neural radiance fields for novel view synthesis.

NeRF based on Jittor could be trained by:

```
bash download_example_data.sh
python nerf.py --config configs/lego.txt
```

The rendering results for synthesized scenes：

<p align="left">
<img src="data/imgs/lego.gif" width="260" style="max-width:50%;">
<img src="data/imgs/hotdog.gif" width="260" style="max-width:50%;">
<img src="data/imgs/mic.gif" width="260" tyle="max-width:50%;">
</p>


The rendering results for real scenes：

<p align="left">
<img src="data/imgs/fern.gif" width="260" style="max-width:50%;">
<img src="data/imgs/flower.gif" width="260" style="max-width:50%;">
<img src="data/imgs/horn.gif" width="260" style="max-width:50%;">
</p>


Our implementation is 1.4 times faster than the official version, and use less GPU memory.


## Citation

Jrender is based on Jittor, and if you use Jrender in your work，please cite:
```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--222103},
  year={2020}
}
```

And if you use the NMR and SoftRas algorithms in Jrender, please cite:
```
@InProceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}

@article{liu2019softras,
  title={Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning},
  author={Liu, Shichen and Li, Tianye and Chen, Weikai and Li, Hao},
  journal={The IEEE International Conference on Computer Vision (ICCV)},
  month = {Oct},
  year={2019}
}
```

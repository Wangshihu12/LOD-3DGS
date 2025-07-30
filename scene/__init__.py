#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, load_iteration=-1, shuffle=True, resolution_scales=[1.0]):
        """
        3D高斯喷射场景初始化函数
        
        功能描述：创建和初始化多层级LOD场景，加载相机数据和高斯模型
        
        参数：
        @param args: 模型参数对象，包含数据路径、球谐阶数等配置信息
        @param load_iteration: 要加载的模型迭代次数，-1表示加载最新的训练模型
        @param shuffle: 是否随机打乱相机顺序，用于训练时的数据增强
        @param resolution_scales: 分辨率缩放比例列表，支持多分辨率训练
        """
        # 设置模型路径和初始化基本属性
        self.model_path = args.model_path
        self.loaded_iter = None     # 实际加载的迭代次数
        self.level = 0              # 当前层级（可能用于调试）
        
        # 检查是否存在已训练的模型，并确定要加载的迭代次数
        if os.path.exists(os.path.join(self.model_path, "point_cloud")):
            if load_iteration == -1:
                # 如果指定-1，则搜索最大迭代次数（最新的模型）
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                # 否则使用指定的迭代次数
                self.loaded_iter = load_iteration
            print("[ Scene ] Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化相机数据字典，支持多分辨率
        self.train_cameras = {}     # 训练相机字典 {分辨率: 相机列表}
        self.test_cameras = {}      # 测试相机字典 {分辨率: 相机列表}
        
        # 根据数据目录结构自动识别场景类型并加载场景信息
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 检测到sparse文件夹，判断是否使用LOD
            if args.use_lod:
                # 使用八叉树LOD数据集
                print("[ Scene ] Found sparse folder, assuming Octree data set!")
                scene_info = sceneLoadTypeCallbacks["Octree"](args.source_path, args.images, args.depths, args.eval)
            else:
                # 使用标准Colmap数据集
                print("[ Scene ] Found sparse folder, assuming Colmap data set!")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # 检测到Blender格式的训练变换文件
            print("[ Scene ] Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            # 检测到Blender格式的通用变换文件
            print("[ Scene ] Found transforms.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            # 无法识别的场景类型，抛出错误
            assert False, "Could not recognize scene type!"

        # 初始化多层级LOD系统
        self.max_level = scene_info.max_level                              # 最大LOD层级数
        self.beta = np.log(self.max_level+1)                              # LOD计算中的衰减参数β
        # 为每个LOD层级创建独立的高斯模型
        self.gaussians = [GaussianModel(args.sh_degree, level) for level in range(self.max_level + 1)]
        
        # 如果没有加载已有模型，则保存输入数据和相机参数
        if not self.loaded_iter:
            # 复制原始点云文件到模型目录
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            
            # 准备相机参数的JSON格式数据
            json_cams = []
            camlist = []
            # 收集所有相机（测试+训练）
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            
            # 将相机参数转换为JSON格式
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            
            # 保存相机参数到JSON文件
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 如果启用了数据打乱，随机重排相机顺序（保证多分辨率一致性）
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # 随机打乱训练相机顺序
            random.shuffle(scene_info.test_cameras)   # 随机打乱测试相机顺序

        # 设置相机范围，用于场景归一化
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 为每个分辨率缩放比例创建相机列表
        for resolution_scale in resolution_scales:
            print("[ Scene ] Loading Training Cameras")
            # 从相机信息创建训练相机对象列表
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("[ Scene ] Loading Test Cameras")
            # 从相机信息创建测试相机对象列表
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 初始化深度范围，用于LOD层级计算
        self.depth_min = torch.tensor(float('inf'))    # 最小深度（初始为正无穷）
        self.depth_max = torch.tensor(-float('inf'))   # 最大深度（初始为负无穷）
        
        # 加载或创建高斯模型
        import time
        st = time.time()  # 开始计时
        
        # 遍历所有LOD层级
        for level in range(self.max_level + 1):
            if self.loaded_iter:
                # 如果存在已训练模型，从文件加载高斯参数
                self.gaussians[level].load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "level_{}.ply".format(level)))
            else:
                # 如果没有已训练模型，从点云数据创建初始高斯模型
                self.gaussians[level].create_from_pcd(scene_info.point_cloud[level], self.cameras_extent)

            # 计算场景的深度范围，遍历所有训练相机
            for cam in self.train_cameras[1.0]:
                # 获取当前层级的高斯点坐标
                xyz = self.gaussians[level].get_xyz.detach()
                # 计算高斯点在当前相机视角下的深度
                depth_z = self.get_z_depth(xyz, cam.world_view_transform)
                # 更新最小深度（确保非负）
                self.depth_min = torch.min(self.depth_min, torch.max(depth_z.min(), torch.tensor(0.0)))
                # 更新最大深度
                self.depth_max = torch.max(self.depth_max, depth_z.max())
        
        # 调整深度范围，添加一定的边距以提高稳定性
        # 将深度范围扩展1.3倍，然后取95%作为最大值，5%作为最小值
        self.depth_max = 0.95 * 1.3 * (self.depth_max - self.depth_min) + self.depth_min
        self.depth_min = 0.05 * 1.3 * (self.depth_max - self.depth_min) + self.depth_min
        
        print("[ Scene ] Initialize scene depth range at [{:2f}, {:2f}]".format(self.depth_min.cpu(), self.depth_max.cpu()))
        
        # 结束计时并输出模型创建耗时
        et = time.time()
        print("[ Scene ] Gaussian Model creation took {} seconds".format(et - st))
        

    def get_z_depth(self, xyz, viewmatrix):
        homogeneous_xyz = torch.cat((xyz, torch.ones(xyz.shape[0], 1, dtype=xyz.dtype, device=xyz.device)), dim=1)
        projected_xyz= torch.matmul(homogeneous_xyz, viewmatrix)
        depth_z = projected_xyz[:,2]
        return depth_z

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if self.max_level == 0:
            self.save_full_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:   
            for level in range(self.max_level+1):
                self.gaussians[level].save_ply(os.path.join(point_cloud_path, "level_{}.ply".format(level)))
        

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.gaussians[-1]._features_dc.shape[1]*self.gaussians[-1]._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.gaussians[-1]._features_rest.shape[1]*self.gaussians[-1]._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.gaussians[-1]._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.gaussians[-1]._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_full_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz, f_dc, f_rest, opacity, scales, rotations = [], [], [], [], [], []
        for level in range(self.max_level + 1):
            xyz.append(self.gaussians[level]._xyz)
            f_dc.append(self.gaussians[level]._features_dc)
            f_rest.append(self.gaussians[level]._features_rest)
            opacity.append(self.gaussians[level]._opacity)
            scales.append(self.gaussians[level]._scaling)
            rotations.append(self.gaussians[level]._rotation)

        xyz = torch.cat(xyz, dim=0).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat(f_dc, dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = torch.cat(f_rest, dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = torch.cat(opacity, dim=0).detach().cpu().numpy()
        scale = torch.cat(scales, dim=0).detach().cpu().numpy()
        rotation = torch.cat(rotations, dim=0).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getGaussians(self, level=-1):
        return self.gaussians[level]
    
    def getLevels(self):
        return self.max_level
    
    def update_max_radii2D(self, radii, visibility_filter, masks):
        level_start = 0
        expanded_visibility_filter = torch.zeros(masks.shape[0], dtype=torch.bool, device=visibility_filter.device)
        expanded_radii = torch.zeros(masks.shape[0], dtype=radii.dtype, device=radii.device)
        expanded_visibility_filter[masks] = visibility_filter
        expanded_radii[masks] = radii
        for level in range(self.max_level + 1):
            level_offset = self.gaussians[level].max_radii2D.shape[0]
            level_radii = expanded_radii[level_start:level_start+level_offset]
            level_visibility_filter = expanded_visibility_filter[level_start:level_start+level_offset]
            self.gaussians[level].max_radii2D[level_visibility_filter] = torch.max(self.gaussians[level].max_radii2D[level_visibility_filter], level_radii[level_visibility_filter])
            level_start += level_offset

    def training_setup(self, args):
        for level in range(self.max_level + 1):
            self.gaussians[level].training_setup(args)

    def restore(self, params, args):
        for level in range(self.max_level + 1):
            self.gaussians[level].restore(params, args)
    
    def update_learning_rate(self, iters):
        for level in range(self.max_level + 1):
            self.gaussians[level].update_learning_rate(iters)
        
    def oneupSHdegree(self):
        for level in range(self.max_level + 1):
            self.gaussians[level].oneupSHdegree()
    
    def add_densification_stats(self, viewspace_point, visibility_filter, masks):
        level_start = 0
        viewspace_point_grad = viewspace_point.grad
        expanded_viewspace_point_grad = torch.zeros(masks.shape[0], 3, dtype=viewspace_point_grad.dtype, device=viewspace_point_grad.device)
        expanded_visibility_filter = torch.zeros(masks.shape[0], dtype=torch.bool, device=visibility_filter.device)
        expanded_viewspace_point_grad[masks,:] = viewspace_point_grad
        expanded_visibility_filter[masks] = visibility_filter
        for level in range(self.max_level + 1):
            level_offset = self.gaussians[level].get_xyz.shape[0]
            level_viewspace_point_grad = expanded_viewspace_point_grad[level_start:level_start + level_offset]
            level_visibility_filter = expanded_visibility_filter[level_start:level_start + level_offset]
            self.gaussians[level].add_densification_stats(level_viewspace_point_grad, level_visibility_filter)    
            level_start += self.gaussians[level].get_xyz.shape[0]

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        for level in range(self.max_level + 1):
            scale = np.min([np.sqrt(2) ** (self.max_level - level), 4.0])   #np.log(self.max_level - level + 1.0) + 1.0
            if max_screen_size:
                max_screen_size = max_screen_size * scale
            self.gaussians[level].densify_and_prune(max_grad * scale, min_opacity, extent * scale, max_screen_size)
    
    def reset_opacity(self):
        for level in range(self.max_level + 1):
            self.gaussians[level].reset_opacity()

    def optimizer_step(self):
        for level in range(self.max_level + 1):
            self.gaussians[level].optimizer.step()
            self.gaussians[level].optimizer.zero_grad(set_to_none = True)


    def get_gaussian_parameters(self, viewpoint, compute_cov3D_python, scaling_modifier=1.0, random=-1):
        """
        获取多层级3D高斯模型参数的函数
        
        功能描述：实现LOD（Level of Detail）系统，根据视角和深度动态选择合适的高斯细节层级
        
        参数：
        @param viewpoint: 视点变换矩阵，用于计算深度和确定激活层级
        @param compute_cov3D_python: 是否在Python中计算3D协方差矩阵的布尔标志
        @param scaling_modifier: 缩放修饰符，用于调整高斯点大小，默认为1.0
        @param random: 层级选择参数，-1表示基于深度自动选择，>=0表示强制使用指定层级
        
        @return: 返回过滤后的高斯参数元组：坐标、特征、不透明度、缩放、旋转、协方差、球谐阶数、掩码
        """

        # 获取所有可用的LOD层级范围（从0到最大层级）
        levels = range(self.max_level + 1)
        
        # 定义属性获取函数：从每个层级的高斯模型中提取指定属性
        get_attrs = lambda attr: [getattr(self.gaussians[level], attr) for level in levels]
        
        # 从所有层级中提取高斯模型的基本参数
        # xyz: 各层级高斯点的3D坐标列表
        # features: 各层级高斯点的特征向量列表（包含颜色信息）
        # opacity: 各层级高斯点的不透明度列表
        # scales: 各层级高斯点的缩放参数列表
        # rotations: 各层级高斯点的旋转参数列表
        xyz, features, opacity, scales, rotations = map(get_attrs, ['get_xyz', 'get_features', 'get_opacity', 'get_scaling', 'get_rotation'])

        # 如果需要在Python中计算3D协方差矩阵，则预计算所有层级的协方差
        # 使用最高层级（-1索引）的高斯模型计算协方差，并复制到所有层级
        cov3D_precomp = [self.gaussians[-1].get_covariance(scaling_modifier)] * len(xyz) if compute_cov3D_python else None

        # 根据'random'参数定义层级激活策略
        if random < 0:
            # 自动LOD模式：基于深度自适应选择层级
            
            # 计算每个层级中所有高斯点在当前视角下的Z深度
            depths = [self.get_z_depth(xyz_lvl.detach(), viewpoint) for xyz_lvl in xyz]
            
            # 计算每个高斯点的激活层级
            # 公式：level = clamp((max_level + 1) * exp(-beta * |depth| / depth_max), 0, max_level)
            # 深度越大，激活层级越低（细节越少）；深度越小，激活层级越高（细节越多）
            act_levels = [torch.clamp((self.max_level + 1) * torch.exp(-1.0 * self.beta * torch.abs(depth) / self.depth_max), 0, self.max_level) for depth in depths]
            
            # 将连续的激活层级向下取整，得到离散的层级索引
            act_levels = [torch.floor(level) for level in act_levels]
            
            # 为每个层级创建布尔过滤器：只有当高斯点的激活层级等于当前层级时才为True
            filters = [act_level == level for act_level, level in zip(act_levels, levels)]
        else:
            # 固定LOD模式：只激活指定的单一层级
            
            # 为每个层级创建过滤器：只有当前层级等于指定的random值时才激活该层级的所有高斯点
            filters = [torch.full_like(xyz[level][:,0], level == random, dtype=torch.bool) for level in levels]

        # 定义张量连接函数：将多个层级的属性张量沿第0维连接
        concat_attrs = lambda attrs: torch.cat(attrs, dim=0)
        
        # 连接所有层级的属性，形成完整的高斯点集合
        xyz, features, opacity, scales, rotations, filters = map(concat_attrs, [xyz, features, opacity, scales, rotations, filters])

        # 定义过滤函数：根据过滤器选择激活的高斯点
        filtered = lambda attr: attr[filters]
        
        # 应用过滤器，只保留激活层级中的高斯点参数
        xyz, features, opacity, scales, rotations = map(filtered, [xyz, features, opacity, scales, rotations])

        # 如果需要预计算协方差矩阵，也对其应用相同的过滤器
        if compute_cov3D_python:
            cov3D_precomp = filtered(concat_attrs(cov3D_precomp))

        # 获取球谐函数的阶数信息（使用最高层级的设置）
        active_sh_degree, max_sh_degree = self.gaussians[-1].active_sh_degree, self.gaussians[-1].max_sh_degree

        # 返回过滤后的高斯模型参数
        return xyz, features, opacity, scales, rotations, cov3D_precomp, active_sh_degree, max_sh_degree, filters

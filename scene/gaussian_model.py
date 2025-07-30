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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import laspy
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:
    """
    3D高斯模型类
    
    功能描述：实现3D高斯喷射的核心模型，包含高斯点的位置、颜色、形状等参数以及相关的训练和优化功能
    """

    def setup_functions(self):
        """
        设置模型中使用的各种激活函数和变换函数
        
        功能描述：定义参数的激活函数，确保参数在合理的数值范围内
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            从缩放和旋转参数构建协方差矩阵
            
            参数：
            @param scaling: 缩放参数张量
            @param scaling_modifier: 缩放修饰符
            @param rotation: 旋转四元数张量
            
            @return: 对称协方差矩阵
            """
            # 构建缩放-旋转矩阵L
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 计算协方差矩阵：C = L * L^T
            actual_covariance = L @ L.transpose(1, 2)
            # 提取对称矩阵的上三角部分（6个独立元素）
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 缩放参数的激活函数：使用指数函数确保缩放值为正
        self.scaling_activation = torch.exp
        # 缩放参数的逆激活函数：对数函数
        self.scaling_inverse_activation = torch.log

        # 协方差矩阵构建函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 不透明度的激活函数：sigmoid函数将值限制在[0,1]范围
        self.opacity_activation = torch.sigmoid
        # 不透明度的逆激活函数：logit函数
        self.inverse_opacity_activation = inverse_sigmoid

        # 旋转四元数的激活函数：归一化确保单位四元数
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, level:int = 0):
        """
        初始化3D高斯模型
        
        参数：
        @param sh_degree: 球谐函数的最大阶数，控制颜色的复杂度
        @param level: LOD层级，用于多层级细节系统
        """
        # 球谐函数相关参数
        self.active_sh_degree = 0           # 当前激活的球谐函数阶数
        self.max_sh_degree = sh_degree      # 最大球谐函数阶数
        self.level = level                  # LOD层级
        
        # 高斯模型的核心参数（使用下划线表示原始参数，需要通过激活函数转换）
        self._xyz = torch.empty(0)          # 3D位置坐标
        self._features_dc = torch.empty(0)  # 球谐函数的DC分量（0阶，基础颜色）
        self._features_rest = torch.empty(0) # 球谐函数的其他分量（1阶及以上）
        self._scaling = torch.empty(0)      # 缩放参数（椭球的三个轴长）
        self._rotation = torch.empty(0)     # 旋转四元数
        self._opacity = torch.empty(0)      # 不透明度
        
        # 训练相关的辅助参数
        self.max_radii2D = torch.empty(0)       # 高斯点在2D屏幕上的最大半径
        self.xyz_gradient_accum = torch.empty(0) # 位置梯度累积（用于密集化）
        self.denom = torch.empty(0)             # 梯度统计的分母
        self.optimizer = None                   # 优化器对象
        self.percent_dense = 0                  # 密集化阈值百分比
        self.spatial_lr_scale = 0              # 空间学习率缩放因子
        
        # 初始化激活函数
        self.setup_functions()
        
    def capture(self):
        """
        捕获模型的当前状态
        
        功能描述：保存模型的所有参数和训练状态，用于检查点保存
        
        @return: 包含所有模型状态的元组
        """
        return (
            self.active_sh_degree,      # 当前球谐函数阶数
            self._xyz,                  # 位置参数
            self._features_dc,          # DC特征
            self._features_rest,        # 其他特征
            self._scaling,              # 缩放参数
            self._rotation,             # 旋转参数
            self._opacity,              # 不透明度参数
            self.max_radii2D,           # 最大2D半径
            self.xyz_gradient_accum,    # 梯度累积
            self.denom,                 # 分母统计
            self.optimizer.state_dict(), # 优化器状态
            self.spatial_lr_scale,      # 空间学习率缩放
        )
    
    def restore(self, model_args, training_args):
        """
        恢复模型状态
        
        功能描述：从保存的状态中恢复模型参数和训练状态
        
        参数：
        @param model_args: 模型参数元组（来自capture方法）
        @param training_args: 训练参数对象
        """
        # 解包模型参数
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        
        # 重新设置训练配置
        self.training_setup(training_args)
        # 恢复梯度统计
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # 恢复优化器状态
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        """
        获取激活后的缩放参数
        
        @return: 经过指数激活的缩放参数（确保为正值）
        """
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        """
        获取激活后的旋转参数
        
        @return: 归一化后的旋转四元数
        """
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        """
        获取3D位置坐标
        
        @return: 高斯点的3D位置张量
        """
        return self._xyz
    
    @property
    def get_features(self):
        """
        获取完整的球谐函数特征
        
        功能描述：将DC分量和其他分量连接为完整的特征向量
        
        @return: 完整的球谐函数特征张量
        """
        features_dc = self._features_dc      # DC分量（0阶）
        features_rest = self._features_rest  # 其他分量（1阶及以上）
        # 在特征维度上连接DC和其他分量
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        """
        获取激活后的不透明度
        
        @return: 经过sigmoid激活的不透明度（范围[0,1]）
        """
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        """
        计算3D协方差矩阵
        
        参数：
        @param scaling_modifier: 缩放修饰符，用于调整高斯椭球大小
        
        @return: 3D协方差矩阵
        """
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """
        增加球谐函数阶数
        
        功能描述：在训练过程中逐步增加球谐函数的复杂度
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        从点云数据创建高斯模型
        
        功能描述：使用初始点云数据初始化3D高斯模型的各项参数
        
        参数：
        @param pcd: 基础点云对象，包含位置和颜色信息
        @param spatial_lr_scale: 空间学习率缩放因子
        """
        self.spatial_lr_scale = spatial_lr_scale
        
        # 将点云位置转换为CUDA张量
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 将RGB颜色转换为球谐函数的DC分量
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # 初始化球谐函数特征张量
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color    # 设置DC分量（0阶）
        features[:, 3:, 1:] = 0.0           # 其他分量初始化为0

        print(f"[ Scene ] Number of points at Level {self.level}: ", fused_point_cloud.shape[0])

        # 计算最近邻距离，用于初始化缩放参数
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 缩放参数基于最近邻距离，取对数形式
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        # 初始化旋转四元数为单位四元数
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1  # w分量为1，表示无旋转

        # 初始化不透明度为较小值（0.1），使用逆sigmoid变换
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将所有参数转换为可训练的神经网络参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # 初始化最大2D半径记录
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """
        设置训练相关的参数和优化器
        
        功能描述：配置不同参数的学习率和优化器，初始化训练统计
        
        参数：
        @param training_args: 训练参数对象，包含各种学习率设置
        """
        # 设置密集化阈值
        self.percent_dense = training_args.percent_dense
        
        # 初始化梯度统计张量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 为不同类型的参数设置不同的学习率
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},  # 高阶特征使用较低学习率
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 创建Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # 设置位置参数的指数衰减学习率调度器
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """
        更新学习率
        
        功能描述：根据训练迭代次数更新位置参数的学习率
        
        参数：
        @param iteration: 当前训练迭代次数
        
        @return: 更新后的学习率
        """
        # 遍历所有参数组，只更新位置参数的学习率
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # 使用指数衰减调度器计算新的学习率
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """
        构建属性名称列表
        
        功能描述：生成PLY文件保存时使用的属性名称列表
        
        @return: 属性名称列表
        """
        # 基础属性：位置和法向量（法向量设为0）
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        # DC特征属性名称
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        
        # 其他球谐函数特征属性名称
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        
        # 不透明度属性
        l.append('opacity')
        
        # 缩放参数属性名称
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        
        # 旋转参数属性名称
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        return l

    def save_ply(self, path):
        """
        将高斯模型保存为PLY文件
        
        功能描述：将模型的所有参数保存到PLY格式文件中
        
        参数：
        @param path: 保存文件的路径
        """
        # 创建目录（如果不存在）
        mkdir_p(os.path.dirname(path))

        # 将所有参数从GPU转移到CPU并转换为numpy数组
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # 法向量设为零向量
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 构建PLY文件的数据类型定义
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 创建结构化数组
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # 连接所有属性数据
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        # 创建PLY元素并写入文件
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def reset_opacity(self):
        """
        重置不透明度参数
        
        功能描述：将所有高斯点的不透明度重置为较小值，防止过度不透明
        """
        # 将不透明度限制在0.01以下，然后应用逆sigmoid变换
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # 更新优化器中的不透明度参数
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        从PLY文件加载高斯模型
        
        功能描述：从保存的PLY文件中恢复模型的所有参数
        
        参数：
        @param path: PLY文件路径
        """
        # 读取PLY文件
        plydata = PlyData.read(path)

        # 提取位置坐标
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # 提取不透明度
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 提取DC特征（RGB的球谐表示）
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 提取其他球谐函数特征
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # 重新整形为(P, F, SH_coeffs except DC)格式
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # 提取缩放参数
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 提取旋转参数
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 将所有数据转换为CUDA张量并设置为可训练参数
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # 重新初始化2D半径记录
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # 设置球谐函数阶数为最大值
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        在优化器中替换指定参数的张量
        
        功能描述：更新优化器中的参数张量，同时保持优化器状态的一致性
        
        参数：
        @param tensor: 新的参数张量
        @param name: 参数名称
        
        @return: 更新后的可优化张量字典
        """
        optimizable_tensors = {}
        
        # 遍历所有参数组，找到匹配的参数
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 获取当前参数的优化器状态
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 重置Adam优化器的动量统计
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 删除旧参数并设置新参数
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        根据掩码剪枝优化器中的参数
        
        功能描述：移除被标记的高斯点，更新优化器状态
        
        参数：
        @param mask: 布尔掩码，True表示保留的点
        
        @return: 剪枝后的可优化张量字典
        """
        optimizable_tensors = {}
        
        # 遍历所有参数组进行剪枝
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 对优化器状态应用掩码
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 更新参数
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果没有优化器状态，直接应用掩码
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        剪枝高斯点
        
        功能描述：移除不需要的高斯点，包括参数和相关统计信息
        
        参数：
        @param mask: 布尔掩码，True表示要移除的点
        """
        # 取反掩码，得到要保留的点
        valid_points_mask = ~mask
        # 剪枝优化器参数
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新所有模型参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 更新梯度统计和其他辅助信息
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新张量连接到优化器参数中
        
        功能描述：添加新的高斯点到现有模型中，更新优化器状态
        
        参数：
        @param tensors_dict: 包含新参数的字典
        
        @return: 更新后的可优化张量字典
        """
        optimizable_tensors = {}
        
        # 遍历所有参数组，添加新的张量
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                # 扩展优化器状态（为新参数添加零初始化的动量）
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                # 连接新参数
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果没有优化器状态，直接连接
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """
        密集化操作的后处理
        
        功能描述：将新创建的高斯点添加到模型中，并重置相关统计信息
        
        参数：
        @param new_xyz: 新的位置参数
        @param new_features_dc: 新的DC特征
        @param new_features_rest: 新的其他特征
        @param new_opacities: 新的不透明度
        @param new_scaling: 新的缩放参数
        @param new_rotation: 新的旋转参数
        """
        # 构建新参数字典
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 将新参数添加到优化器中
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # 更新模型参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 重置所有高斯点的梯度统计和2D半径记录
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        密集化：分割大的高斯点
        
        功能描述：将大尺寸且梯度大的高斯点分割为多个小高斯点
        
        参数：
        @param grads: 位置梯度
        @param grad_threshold: 梯度阈值
        @param scene_extent: 场景范围
        @param N: 分割数量，默认为2
        """
        n_init_points = self.get_xyz.shape[0]
        
        # 创建梯度填充数组（处理梯度数组长度不一致的情况）
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # 选择满足条件的点：梯度大且尺寸大
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 为选中的点生成新的位置
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)              # 使用原始缩放作为标准差
        means =torch.zeros((stds.size(0), 3),device="cuda")                # 零均值
        samples = torch.normal(mean=means, std=stds)                       # 高斯采样
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)  # 旋转矩阵
        # 将采样点旋转并平移到原始位置附近
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # 缩小新高斯点的尺寸
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 复制其他参数
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # 添加新的高斯点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 移除原始的大高斯点
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        密集化：克隆小的高斯点
        
        功能描述：复制小尺寸但梯度大的高斯点
        
        参数：
        @param grads: 位置梯度
        @param grad_threshold: 梯度阈值
        @param scene_extent: 场景范围
        """
        # 选择满足条件的点：梯度大但尺寸小
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 直接复制选中点的所有参数
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 添加克隆的高斯点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        执行密集化和剪枝操作
        
        功能描述：组合密集化（克隆和分割）和剪枝操作来优化高斯点分布
        
        参数：
        @param max_grad: 最大梯度阈值
        @param min_opacity: 最小不透明度阈值
        @param extent: 场景范围
        @param max_screen_size: 最大屏幕尺寸
        """
        # 计算平均梯度
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0  # 处理NaN值

        # 执行密集化操作
        self.densify_and_clone(grads, max_grad, extent)  # 克隆小高斯点
        self.densify_and_split(grads, max_grad, extent)  # 分割大高斯点

        # 构建剪枝掩码：移除不透明度过小的点
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if max_screen_size:
            # 移除屏幕上过大的点和世界空间中过大的点
            big_points_vs = self.max_radii2D > max_screen_size                    # 屏幕空间过大
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent    # 世界空间过大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 执行剪枝
        self.prune_points(prune_mask)

        # 清空CUDA缓存
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        添加密集化统计信息
        
        功能描述：累积高斯点在屏幕空间的梯度信息，用于后续的密集化决策
        
        参数：
        @param viewspace_point_tensor: 屏幕空间点张量
        @param update_filter: 更新过滤器，标识哪些点需要更新统计
        """
        # 累积屏幕空间2D梯度的模长
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        # 累积统计次数
        self.denom[update_filter] += 1
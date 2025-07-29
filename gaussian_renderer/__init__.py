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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh

def render(viewpoint_camera, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, 
        pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, cov3D_precomp = None, colors_precomp = None ):
    """
    3D高斯喷射场景渲染函数
    
    功能描述：将3D高斯点云从世界坐标系投影到2D图像平面，生成渲染图像
    
    参数：
    @param viewpoint_camera: 视点相机对象，包含相机内外参数、视角变换矩阵等
    @param xyz: 3D高斯中心点坐标张量，形状为 [N, 3]
    @param features: 高斯特征向量，通常包含球谐系数用于颜色计算，形状为 [N, feature_dim, 3]
    @param opacity: 高斯点的不透明度，形状为 [N, 1]
    @param scales: 高斯椭球的缩放参数，形状为 [N, 3]
    @param rotations: 高斯椭球的旋转四元数，形状为 [N, 4]
    @param active_sh_degree: 当前激活的球谐函数阶数
    @param max_sh_degree: 最大球谐函数阶数
    @param pipe: 渲染管线参数对象，包含各种渲染配置
    @param bg_color: 背景颜色张量，必须在GPU上
    @param scaling_modifier: 缩放修饰符，用于调整高斯点大小，默认为1.0
    @param override_color: 覆盖颜色，如果提供则直接使用而不计算球谐颜色
    @param cov3D_precomp: 预计算的3D协方差矩阵，可选
    @param colors_precomp: 预计算的颜色，可选
    
    @return: 包含渲染结果的字典，包含渲染图像、深度图、可见性过滤器等
    """
 
    # 创建屏幕空间点坐标的零张量，用于PyTorch计算2D屏幕坐标的梯度
    # requires_grad=True 确保可以对屏幕空间坐标计算梯度，这对训练优化很重要
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 保留梯度信息，确保反向传播时能够计算屏幕空间点的梯度
        screenspace_points.retain_grad()
    except:
        # 如果保留梯度失败则忽略（可能在推理模式下）
        pass

    # 设置光栅化配置参数
    # 计算相机视场角的正切值，用于透视投影
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 水平视场角的一半的正切值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 垂直视场角的一半的正切值

    # 创建高斯光栅化设置对象，包含所有渲染所需的配置参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),        # 输出图像高度
        image_width=int(viewpoint_camera.image_width),          # 输出图像宽度
        tanfovx=tanfovx,                                        # 水平视场角正切值
        tanfovy=tanfovy,                                        # 垂直视场角正切值
        bg=bg_color,                                            # 背景颜色
        scale_modifier=scaling_modifier,                        # 缩放修饰符，用于调整高斯点大小
        viewmatrix=viewpoint_camera.world_view_transform,       # 世界到视图的变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,        # 完整的投影变换矩阵
        sh_degree=active_sh_degree,                             # 当前使用的球谐函数阶数
        campos=viewpoint_camera.camera_center,                  # 相机中心位置
        prefiltered=False,                                      # 是否预过滤，False表示不预过滤
        debug=pipe.debug                                        # 是否开启调试模式
    )

    # 创建高斯光栅化器对象，用于执行实际的渲染操作
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 准备渲染所需的基本参数
    means3D = xyz                    # 3D高斯中心点坐标
    means2D = screenspace_points     # 2D屏幕空间点坐标（初始为零，将由光栅化器计算）
    opacity = opacity                # 不透明度参数

    # 颜色处理：根据不同情况处理颜色信息
    # 如果提供了预计算颜色则使用，否则从球谐系数计算，或由光栅化器处理SH->RGB转换
    shs = None                       # 球谐系数
    colors_precomp = None           # 预计算颜色
    
    if override_color is None:
        # 如果没有提供覆盖颜色，则需要处理球谐系数
        if pipe.convert_SHs_python:
            # 在Python中进行球谐函数到RGB的转换
            # 重新排列特征张量维度以匹配球谐函数计算要求
            shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            
            # 计算从高斯点到相机的方向向量
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            # 归一化方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            # 使用球谐函数计算RGB颜色
            sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            # 将颜色值限制在合理范围内（添加0.5偏移并确保非负）
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 将球谐系数传递给光栅化器，由CUDA代码进行SH->RGB转换
            shs = features
    else:
        # 如果提供了覆盖颜色，直接使用
        colors_precomp = override_color

    # 注释：如果提供了预计算的3D协方差矩阵则使用，否则光栅化器将从缩放和旋转参数计算

    # 执行高斯光栅化：将可见的高斯点渲染到图像上，并获取它们在屏幕上的半径
    rendered_image, depth, radii = rasterizer(
        means3D = means3D,              # 3D高斯中心点坐标
        means2D = means2D,              # 2D屏幕空间坐标（由光栅化器更新）
        shs = shs,                      # 球谐系数（如果在GPU中转换）
        colors_precomp = colors_precomp, # 预计算的颜色（如果在Python中转换）
        opacities = opacity,            # 不透明度
        scales = scales,                # 缩放参数
        rotations = rotations,          # 旋转四元数
        cov3D_precomp = cov3D_precomp) # 预计算的3D协方差矩阵（可选）

    # 那些被视锥体剔除或半径为0的高斯点是不可见的
    # 它们将从用于分割标准的数值更新中排除
    return {"render": rendered_image,           # 渲染得到的RGB图像
            "depth": depth,                     # 深度图
            "viewspace_points": screenspace_points,  # 屏幕空间点坐标（包含梯度信息）
            "visibility_filter" : radii > 0,   # 可见性过滤器，标识哪些高斯点在屏幕上可见
            "radii": radii}                     # 各高斯点在屏幕上的半径

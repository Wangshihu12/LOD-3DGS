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
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

def render_set(model_path, name, iteration, views, scene, pipeline, background):
    """
    渲染指定视图集合的函数
    
    功能描述：对给定的视图集合进行3D高斯模型渲染，生成渲染图像和对应的真实图像
    
    参数：
    @param model_path: 模型路径，用于保存渲染结果的根目录
    @param name: 数据集名称（如 "train" 或 "test"）
    @param iteration: 训练迭代次数，用于标识模型版本
    @param views: 视图列表，包含需要渲染的相机视角
    @param scene: 场景对象，包含3D高斯模型和相关参数
    @param pipeline: 渲染管线参数，控制渲染过程的配置
    @param background: 背景颜色张量
    """
    # 构建渲染结果保存路径：model_path/name/ours_iteration/renders
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # 构建真实图像保存路径：model_path/name/ours_iteration/gt  
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 创建保存目录，如果目录已存在则不报错
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历所有视图进行渲染，使用tqdm显示进度条
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 从场景中获取高斯模型参数
        # xyz: 高斯中心位置坐标
        # features: 特征向量（包含颜色信息）
        # opacity: 不透明度
        # scales: 缩放参数
        # rotations: 旋转参数
        # cov3D_precomp: 预计算的3D协方差矩阵
        # active_sh_degree: 当前激活的球谐函数阶数
        # max_sh_degree: 最大球谐函数阶数
        # masks: 掩码信息
        xyz, features, opacity, scales, rotations, cov3D_precomp, \
            active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(view.world_view_transform, pipeline.compute_cov3D_python)

        # 执行渲染操作，生成当前视角的渲染图像
        rendering = render(view, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, pipeline, background, cov3D_precomp = cov3D_precomp)["render"]
        # 获取真实图像的前三个通道（RGB）
        gt = view.original_image[0:3, :, :]
        # 保存渲染结果为PNG图像，文件名格式为00000.png, 00001.png等
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # 保存对应的真实图像
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    """
    渲染数据集的主函数
    
    功能描述：加载指定迭代次数的3D高斯模型，对训练集和测试集进行渲染
    
    参数：
    @param dataset: 模型参数对象，包含数据集路径、背景设置等配置信息
    @param iteration: 要加载的模型迭代次数，-1表示加载最新的模型
    @param pipeline: 渲染管线参数对象，包含渲染相关的配置
    @param skip_train: 是否跳过训练集渲染的布尔标志
    @param skip_test: 是否跳过测试集渲染的布尔标志
    """
    # 禁用梯度计算，节省内存并加速推理过程
    with torch.no_grad():
        # 注释掉的代码：旧版本的创建方式，需要显式创建高斯模型
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # 创建场景对象，自动加载指定迭代次数的模型权重
        # load_iteration: 指定加载的模型迭代次数
        # shuffle=False: 不打乱数据顺序，保持一致性便于比较
        scene = Scene(dataset, load_iteration=iteration, shuffle=False)

        # 根据数据集配置设置背景颜色
        # 白色背景：[1,1,1] (RGB值为1表示白色)
        # 黑色背景：[0,0,0] (RGB值为0表示黑色)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # 将背景颜色转换为CUDA张量，用于GPU加速渲染
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 如果未设置跳过训练集标志，则渲染训练集
        if not skip_train:
             # 调用render_set函数渲染训练集
             # "train": 数据集类型标识
             # scene.loaded_iter: 实际加载的模型迭代次数
             # scene.getTrainCameras(): 获取训练集的相机视角列表
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, pipeline, background)

        # 如果未设置跳过测试集标志，则渲染测试集
        if not skip_test:
             # 调用render_set函数渲染测试集
             # "test": 数据集类型标识
             # scene.getTestCameras(): 获取测试集的相机视角列表
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("[ INFO ] Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
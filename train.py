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
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, warpped_depth, unwarpped_depth
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    3D高斯喷射模型的训练主函数
    
    功能描述：执行3D高斯模型的完整训练流程，包括场景初始化、渲染、损失计算、优化等
    
    参数：
    @param dataset: 数据集参数对象，包含数据路径和配置信息
    @param opt: 优化器参数对象，包含学习率、迭代次数等训练配置
    @param pipe: 渲染管线参数对象，控制渲染过程的各种设置
    @param testing_iterations: 测试迭代次数列表，指定在哪些迭代进行测试
    @param saving_iterations: 保存迭代次数列表，指定在哪些迭代保存模型
    @param checkpoint_iterations: 检查点迭代次数列表
    @param checkpoint: 检查点文件路径，用于恢复训练
    @param debug_from: 开始调试的迭代次数
    """
    # 初始化起始迭代次数
    first_iter = 0

    # 设置输出文件夹路径
    if not dataset.model_path:
        # 如果在集群环境中运行，使用作业ID作为唯一标识
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            # 否则生成随机UUID作为唯一标识
            unique_str = str(uuid.uuid4())
        # 构建模型保存路径
        dataset.model_path = os.path.join(dataset.source_path, "3D-Gaussian-Splatting", unique_str[0:10])
    print("[ Training ] Output Folder: {}".format(dataset.model_path))
    # 创建模型保存目录
    os.makedirs(dataset.model_path, exist_ok = True)

    # 加载数据集并创建场景对象
    scene = Scene(dataset)
    # 设置训练相关参数（优化器、学习率调度器等）
    scene.training_setup(opt)

    # 提取场景深度最大值，用于深度损失计算
    dataset.depth_max = scene.depth_max.cpu().item()

    # 准备日志记录器和参数提取
    tb_writer = prepare_output_and_logger(dataset)

    # 如果提供了检查点，则从检查点恢复训练状态
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        scene.restore(model_params, opt)

    # 根据数据集配置设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # 创建CUDA背景张量用于GPU渲染
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 初始化视点栈和层级栈
    viewpoint_stack = None  # 存储训练相机视点的栈
    level_stack = None      # 存储LOD层级的栈
    ema_loss_for_log = 0.0  # 用于日志记录的指数移动平均损失
    
    # 创建训练进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):        
        # 尝试连接网络GUI（用于实时可视化）
        if network_gui.conn == None:
            network_gui.try_connect()
        
        # 处理网络GUI连接和交互
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 从GUI接收参数：自定义相机、训练标志、渲染配置等
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # 使用自定义相机进行渲染（用于GUI显示）
                    # 获取高斯模型参数
                    xyz, features, opacity, scales, rotations, cov3D_precomp, \
                        active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(viewpoint_cam.world_view_transform, pipe.compute_cov3D_python, scaling_modifer)
                    # 执行渲染
                    net_image = render(custom_cam,  xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, pipe, background, scaling_modifer, cov3D_precomp = cov3D_precomp)
                    # 将渲染结果转换为字节格式用于网络传输
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                # 发送渲染结果到GUI
                network_gui.send(net_image_bytes, dataset.source_path)
                # 如果需要继续训练且未达到最大迭代次数，跳出GUI循环
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                # 连接异常时断开连接
                network_gui.conn = None

        # 开始计时当前迭代
        iter_start.record()

        # 更新学习率（基于迭代次数的学习率调度）
        scene.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数的阶数，直到达到最大阶数
        if iteration % 1000 == 0:
            scene.oneupSHdegree()

        # 随机选择一个训练相机视点
        if not viewpoint_stack:
            # 如果视点栈为空，重新填充所有训练相机
            viewpoint_stack = scene.getTrainCameras().copy()
        # 随机弹出一个视点
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 随机选择一个LOD层级
        if not level_stack:
            # 如果层级栈为空，重新填充所有可用层级
            level_stack = list(range(-scene.max_level, scene.max_level + 1))
        # 随机选择一个层级
        random_level = level_stack.pop(randint(0, len(level_stack)-1))
        # random_level =-1  # 注释掉的固定层级设置
  
        # 渲染阶段
        if (iteration - 1) == debug_from:
            # 在指定迭代开启调试模式
            pipe.debug = True
        
        # 获取当前视点和层级的高斯模型参数
        # xyz: 高斯中心坐标
        # features: 特征向量（颜色信息）
        # opacity: 不透明度
        # scales: 缩放参数
        # rotations: 旋转参数
        # cov3D_precomp: 预计算的3D协方差矩阵
        # active_sh_degree: 当前激活的球谐函数阶数
        # max_sh_degree: 最大球谐函数阶数
        # masks: 掩码信息
        xyz, features, opacity, scales, rotations, cov3D_precomp, \
            active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(viewpoint_cam.world_view_transform, pipe.compute_cov3D_python, random = random_level)
        
        # 执行渲染操作
        render_pkg = render(viewpoint_cam,  xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, pipe, background, cov3D_precomp = cov3D_precomp)
        
        # 提取渲染结果的各个组件
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        # 对深度图进行后处理
        depth = warpped_depth(depth)
        
        # 损失计算
        # 获取真实图像并移动到GPU
        gt_image = viewpoint_cam.original_image.cuda()
        # 计算L1损失
        Ll1 = l1_loss(image, gt_image)
        # 计算RGB损失：结合L1损失和SSIM损失
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # 初始化深度损失
        depth_loss = 0.0

        # 如果存在深度真值，计算深度损失
        if viewpoint_cam.depth is not None:
            # 获取真实深度图和深度掩码
            gt_depth = viewpoint_cam.depth.cuda()
            gt_depth_mask = viewpoint_cam.depth_mask.cuda()

            # 在训练后期（超过1/3迭代）应用深度正则化策略
            if iteration > opt.iterations / 3:
                # 对无效深度区域填充固定值0.75
                gt_depth = gt_depth * gt_depth_mask + (1 - gt_depth_mask) *  0.75
                # 更新深度掩码：只在渲染深度小于真实深度的区域计算损失
                gt_depth_mask = depth < gt_depth

            # 计算深度L1损失，只在有效掩码区域计算
            depth_loss = 2.0 * l1_loss(depth * gt_depth_mask, gt_depth * gt_depth_mask)
   
        # 总损失 = RGB损失 + 深度损失
        loss = rgb_loss + depth_loss
        # 反向传播计算梯度
        loss.backward()

        # 结束当前迭代计时
        iter_end.record()

        # 在无梯度模式下进行日志记录和其他操作
        with torch.no_grad():
            # 更新进度条显示
            # 使用指数移动平均平滑RGB损失显示
            ema_loss_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_loss_for_log
            ema_depth_loss_for_log = 0.0
            # 如果存在深度损失，也进行指数移动平均
            if viewpoint_cam.depth is not None:
                ema_depth_loss_for_log = 0.4 * depth_loss.item()  + 0.6 * ema_depth_loss_for_log
            
            # 每10次迭代更新一次进度条显示
            if iteration % 10 == 0:
                progress_bar.set_postfix({"RGB Loss": f"{ema_loss_for_log:.{4}f}", "Depth Loss": f"{ema_depth_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            # 训练完成时关闭进度条
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练日志和保存模型
            training_report(tb_writer, iteration, rgb_loss, depth_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            # 在指定迭代次数保存模型
            if (iteration in saving_iterations):
                print("[ Training ] [ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密集化处理（在指定迭代范围内）
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中的最大半径，用于剪枝
                scene.update_max_radii2D(radii, visibility_filter, masks)
                # 累积密集化统计信息
                scene.add_densification_stats(viewspace_point_tensor, visibility_filter, masks)

                # 在指定间隔进行密集化和剪枝操作
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 根据迭代次数设置大小阈值
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 执行密集化和剪枝：基于梯度阈值、不透明度阈值、场景范围和大小阈值
                    scene.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # 定期重置不透明度或在白色背景的特定时机重置
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    scene.reset_opacity()

            # 执行优化器步骤（参数更新）
            if iteration < opt.iterations:
                scene.optimizer_step()

def prepare_output_and_logger(args):    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[ Training ] Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, rgb_loss, depth_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    """
    训练报告函数
    
    功能描述：记录训练日志、执行模型验证、计算性能指标并输出到TensorBoard
    
    参数：
    @param tb_writer: TensorBoard写入器，用于记录训练日志和可视化结果
    @param iteration: 当前训练迭代次数
    @param rgb_loss: RGB颜色损失值
    @param depth_loss: 深度损失值（可能是张量或标量）
    @param l1_loss: L1损失函数
    @param elapsed: 当前迭代的耗时（毫秒）
    @param testing_iterations: 测试迭代次数列表，指定在哪些迭代进行验证
    @param scene: 场景对象，包含高斯模型和相机信息
    @param renderFunc: 渲染函数，用于生成图像
    @param renderArgs: 渲染函数的参数
    """
    # 如果TensorBoard写入器存在，记录训练损失和系统信息
    if tb_writer:
        # 记录RGB损失到TensorBoard
        tb_writer.add_scalar('train_loss_patches/rgb_loss', rgb_loss.item(), iteration)
        
        # 处理深度损失：可能是张量或标量
        if isinstance(depth_loss, torch.Tensor):
            # 如果深度损失是张量，提取其数值
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', rgb_loss.item() + depth_loss.item(), iteration)
        else:
            # 如果深度损失是标量，直接使用
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss, iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', rgb_loss.item() + depth_loss, iteration)
        
        # 记录迭代耗时
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        # 记录GPU内存使用情况（转换为GB单位）
        tb_writer.add_scalar('memory/memory_allocated', torch.cuda.memory_allocated('cuda') / (1024 ** 3), iteration)
        tb_writer.add_scalar('memory/memory_reserved', torch.cuda.memory_reserved('cuda') / (1024 ** 3), iteration)

    # 在指定的测试迭代次数进行模型验证和评估
    if iteration in testing_iterations:
        # 清空CUDA缓存，释放内存
        torch.cuda.empty_cache()
        
        # 配置验证数据集：测试集和训练集样本
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()},  # 测试集：使用所有测试相机
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}  # 训练集：每隔5个相机取样本（第5,10,15,20,25个）
        )

        # 遍历验证配置（测试集和训练集样本）
        for config in validation_configs:
            # 检查相机列表是否非空
            if config['cameras'] and len(config['cameras']) > 0:
                # 初始化结果存储张量
                images = torch.tensor([], device="cuda")    # 渲染图像集合
                gts = torch.tensor([], device="cuda")       # 真实图像集合
                l1_test, psnr_test = [], []                 # L1损失和PSNR指标列表
                
                # 遍历当前配置的所有相机视点
                for idx, viewpoint in enumerate(config['cameras']):
                    # 获取当前视点的高斯模型参数
                    xyz, features, opacity, scales, rotations, cov3D_precomp, \
                        active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(
                            viewpoint.world_view_transform, 
                            renderArgs[0].compute_cov3D_python
                        )
                    
                    # 执行渲染操作
                    results = renderFunc(viewpoint, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, cov3D_precomp = cov3D_precomp, *renderArgs)
                    
                    # 提取并限制渲染图像的像素值范围到[0,1]
                    image = torch.clamp(results["render"], 0.0, 1.0)
                    
                    # 处理深度图：应用深度包装函数并归一化到[0,1]
                    depth = warpped_depth(results["depth"])
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    
                    # 获取并限制真实图像的像素值范围到[0,1]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # 如果存在真实深度图，进行归一化处理
                    if viewpoint.depth is not None:
                        gt_depth = viewpoint.depth.to("cuda")
                        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
                    
                    # 计算性能指标
                    l1_test.append(l1_loss(image, gt_image))        # L1损失
                    psnr_test.append(psnr(image, gt_image).mean())  # PSNR（峰值信噪比）
                    
                    # 记录第一个视点的图像到TensorBoard（用于可视化）
                    if tb_writer and (idx == 0):
                        # 记录渲染结果
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image.unsqueeze(0), global_step=iteration)
                        # 记录深度图
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth.unsqueeze(0), global_step=iteration)
                        
                        # 在第一次测试迭代时记录真实图像（避免重复记录）
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image.unsqueeze(0), global_step=iteration)
                            # 如果存在真实深度图，也记录下来
                            if viewpoint.depth is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth_depth".format(viewpoint.image_name), gt_depth.unsqueeze(0), global_step=iteration)
                
                # 记录不同LOD（Level of Detail）层级的渲染结果
                for level in range(scene.max_level + 1):
                    # 使用第一个相机视点测试不同层级
                    viewpoint = config['cameras'][0]  #[randint(0, len(config['cameras'])-1)]
                    
                    # 获取指定层级的高斯模型参数
                    xyz, features, opacity, scales, rotations, cov3D_precomp, \
                        active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(
                            viewpoint.world_view_transform, 
                            renderArgs[0].compute_cov3D_python, 
                            random=level  # 指定LOD层级
                        )
                    
                    # 渲染指定层级的图像
                    image = torch.clamp(renderFunc(viewpoint, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, cov3D_precomp = cov3D_precomp, *renderArgs)["render"], 0.0, 1.0)
                    
                    # 记录不同层级的渲染结果到TensorBoard
                    if tb_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/level_{}".format(viewpoint.image_name, level), image.unsqueeze(0), global_step=iteration)

                # 计算平均性能指标
                l1_test = sum(l1_test) / len(l1_test)           # 平均L1损失
                psnr_test = sum(psnr_test) / len(psnr_test)     # 平均PSNR
                
                # 打印当前配置的评估结果
                print("\n[ Training ] [ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                # 记录平均指标到TensorBoard
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 记录模型统计信息到TensorBoard
        if tb_writer:
            # 记录高斯点不透明度的直方图分布
            tb_writer.add_histogram("scene/opacity_histogram", scene.getGaussians().get_opacity, iteration)
            # 记录当前高斯点的总数量
            tb_writer.add_scalar('total_points', scene.getGaussians().get_xyz.shape[0], iteration)
        
        # 再次清空CUDA缓存，释放验证过程中使用的内存
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50, 100, 500, 1_000, 5_000, 10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 10_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("[ Training ] Optimizing With Parameters: " + str(vars(args)))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("[ Training ] Training complete.")
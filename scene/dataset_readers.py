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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, OctreeGaussian, OctreeGaussianNode
import numpy as np
import json
import laspy
import subprocess
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.system_utils import mkdir_p
from scene.gaussian_model import BasicPointCloud
from scene.octree_loader import loadOctree
import queue

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    depth_path: str
    depth: np.array
    width: int
    height: int
    cx: float
    cy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    max_level: int = 16
    depth_min: float = 0.01
    depth_max: float = 1000

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder = None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[ Scene ] Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        cx = width / 2.0
        cy = height / 2.0

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "[ ERROR ] Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path)

        if depths_folder is not None:
            depth_path = os.path.join(depths_folder, image_name + ".png")
            depth = Image.open(depth_path)
        else:
            depth_path = None
            depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, depth_path=depth_path, depth = depth, width=width, height=height, cx = cx, cy = cy)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if vertices.__contains__('red'):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.zeros_like(positions)

    if vertices.__contains__('nx'):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def storeLas(path, xyz, rgb):

    # 1. Create a Las
    header = laspy.LasHeader(point_format=2, version="1.4")
    out_las = laspy.LasData(header)

    # 2. Fill the Las
    out_las.x = xyz[:, 0]
    out_las.y = xyz[:, 1]
    out_las.z = xyz[:, 2]

    def normalize_color(color):
        return (color * 255).astype(np.uint16)

    out_las.red = normalize_color(rgb[:, 0])
    out_las.green = normalize_color(rgb[:, 1])
    out_las.blue = normalize_color(rgb[:, 2])

    # Save the LAS file
    out_las.write(path)

def readColmapSceneInfo(path, images, depths, eval, llffhold=8):
    """
    读取Colmap场景信息的函数
    
    功能描述：从Colmap SfM重建结果中读取相机参数、图像数据和3D点云，构建场景信息
    
    参数：
    @param path: Colmap数据集的根路径
    @param images: 图像文件夹名称，None时使用默认的"images"
    @param depths: 深度图文件夹名称，空字符串表示无深度数据
    @param eval: 是否为评估模式，True时会分割训练集和测试集
    @param llffhold: LLFF风格的数据分割参数，每隔llffhold张图像作为测试集
    
    @return: SceneInfo对象，包含点云、相机信息和归一化参数
    """
    # 尝试读取二进制格式的Colmap输出文件（优先选择，因为加载更快）
    try:
        # Colmap稀疏重建结果的标准路径
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")    # 相机外参文件（位置和朝向）
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")   # 相机内参文件（焦距、主点等）
        
        # 读取二进制格式的相机参数
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)  # 外参：旋转矩阵、平移向量
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)  # 内参：焦距、畸变参数等
    except:
        # 如果二进制文件不存在或读取失败，尝试读取文本格式文件
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        
        # 读取文本格式的相机参数
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 确定图像文件夹路径
    images_dir = "images" if images == None else images
    
    # 读取所有相机信息，包括图像和可选的深度数据
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,                                    # 相机外参数据
        cam_intrinsics=cam_intrinsics,                                    # 相机内参数据
        images_folder=os.path.join(path, images_dir),                     # 图像文件夹路径
        depths_folder=os.path.join(path, depths) if depths != "" else None  # 深度图文件夹路径（可选）
    )
    
    # 按图像名称排序，确保处理顺序的一致性
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 根据eval参数决定是否分割训练集和测试集
    if eval:
        # 评估模式：使用LLFF风格的数据分割策略
        # 每隔llffhold张图像选择一张作为测试集，其余作为训练集
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]  # 训练集：非llffhold倍数的索引
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]   # 测试集：llffhold倍数的索引
    else:
        # 训练模式：所有图像都用于训练，无测试集
        train_cam_infos = cam_infos
        test_cam_infos = []

    # 计算NeRF++风格的场景归一化参数
    # 用于将场景坐标标准化到合适的范围，提高训练稳定性
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 处理3D点云数据：确定点云文件路径
    ply_path = os.path.join(path, "sparse/0/points3D.ply")    # PLY格式点云文件
    bin_path = os.path.join(path, "sparse/0/points3D.bin")    # 二进制格式点云文件
    txt_path = os.path.join(path, "sparse/0/points3D.txt")    # 文本格式点云文件
    
    # 检查PLY文件是否存在，如果不存在则从Colmap原始格式转换
    if not os.path.exists(ply_path):
        print("[ Dataloader ] Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            # 尝试读取二进制格式的3D点云数据
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            # 如果二进制文件读取失败，尝试读取文本格式
            xyz, rgb, _ = read_points3D_text(txt_path)
        
        # 将点云数据保存为PLY格式，便于后续快速加载
        storePly(ply_path, xyz, rgb)
    else:
        # 如果PLY文件已存在，直接加载
        print("[ Dataloader ] Found .ply point cloud, skipping conversion.")
        pcd = fetchPly(ply_path)
    
    # 将点云数据包装为列表格式（兼容多层级LOD系统）
    pcds = [pcd]
    
    # 创建并返回场景信息对象
    scene_info = SceneInfo(
        point_cloud=pcds,                      # 点云数据列表（单层级）
        train_cameras=train_cam_infos,         # 训练相机信息列表
        test_cameras=test_cam_infos,           # 测试相机信息列表
        nerf_normalization=nerf_normalization, # 场景归一化参数
        ply_path=ply_path,                     # 点云文件路径
        max_level=0                            # 最大LOD层级（Colmap数据为0，即单层级）
    )
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        if "aabb" in contents:
            aabb = contents["aabb"]
            scale = max(0.000001,max(max(abs(float(aabb[1][0])-float(aabb[0][0])),
                            abs(float(aabb[1][1])-float(aabb[0][1]))),
                            abs(float(aabb[1][2])-float(aabb[0][2]))))

            offset = [((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) - 0.5 * scale,
                ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) - 0.5 * scale, 
                ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5)- 0.5 * scale]

        elif "scale" in contents and "offset" in contents:
            scale = 1.0 / contents["scale"]
            offset = -np.array(contents["offset"]) * scale
        else:
            scale = 2.6 # default scale for NeRF scenes
            offset = -1.3

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, unit=" images", desc=f"Loading Images")):
            cam_name = os.path.join(path, frame["file_path"])
            if not os.path.exists(cam_name):
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if "cx" in contents and "cy" in contents:
                cx = contents["cx"]
                cy = contents["cy"]
            else:
                cx = image.size[0] / 2
                cy = image.size[1] / 2
                
            # Extract focal length the transform matrix
            if 'camera_angle_x' in contents or 'camera_angle_y' in contents:
                # blender, assert in radians. already downscaled since we use H/W
                fl_x = fov2focal(contents["camera_angle_x"], 2 * cx) if 'camera_angle_x' in contents else None
                fl_y = fov2focal(contents["camera_angle_y"], 2 * cy) if 'camera_angle_y' in contents else None
                if fl_x is None: fl_x = fl_y
                if fl_y is None: fl_y = fl_x
                FovX = fovx = focal2fov(fl_x, 2 * cx)
                FovY = fovy = focal2fov(fl_y, 2 * cy)
            elif 'fl_x' in contents or 'fl_y' in contents:
                fl_x = (contents['fl_x'] if 'fl_x' in contents else contents['fl_y'])
                fl_y = (contents['fl_y'] if 'fl_y' in contents else contents['fl_x'])
                FovX = fovx = focal2fov(fl_x, 2 * cx)
                FovY = fovy = focal2fov(fl_y, 2 * cy)
            elif 'K' in frame or 'intrinsic_matrix' in frame:
                K = frame['K'] if 'K' in frame else frame['intrinsic_matrix']
                FovX = fovx = focal2fov(K[0][0], 2.0*K[0][2])
                FovY = fovy = focal2fov(K[1][1], 2.0*K[1][2])
                cx, cy = K[0][2], K[1][2]
            elif 'focal_length' in frame:
                FovX = fovx = focal2fov(frame['focal_length'], 2 * cx)
                FovY = fovy = focal2fov(frame['focal_length'], 2 * cy)
            else:
                raise Exception("[ ERROR ] No camera intrinsics found in the transforms file.")
                
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], cx = cx, cy = cy))
            
    return cam_infos, scale, offset

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # Read Train Cameras
    if os.path.exists(os.path.join(path, "transforms_train.json")):
        print("[ Dataloader ] Reading Training Transforms From: transforms_train.json")
        train_cam_infos, scale, offset = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    elif os.path.exists(os.path.join(path, "transforms.json")):
        print("[ Dataloader ] Reading Training Transforms From: transforms.json")
        train_cam_infos, scale, offset = readCamerasFromTransforms(path, "transforms.json", white_background, extension)

    # Read Test Cameras
    if os.path.exists(os.path.join(path, "transforms_test.json")) and eval:
        print("[ Dataloader ] Reading Test Transforms From: transforms_test.json")
        test_cam_infos, scale, offset = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Read Point Cloud
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"[ Dataloader ] Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * scale + offset
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    pcds = [pcd]
    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           max_level=0)
    return scene_info

"""function to save the octree class into ply file and store into disk"""

# gaussianmodels need to be replace to real gaussianmodesl
# now the position is aligned with colmap coordinate, do not need to add any offset

def collect_position_buffers(node, level, max_level=16):
    position_buffers = []
    color_buffers = []
    if level <= max_level:
        position_buffer = node.pointcloud.points
        position_buffers.append(position_buffer)
        color_buffer = node.pointcloud.colors
        color_buffers.append(color_buffer)
        if hasattr(node, 'children'):
            for child in node.children:
                if child is not None:
                    result = collect_position_buffers(child, level + 1, max_level)
                    position_buffers.extend(result[0])
                    color_buffers.extend(result[1])
    return position_buffers, color_buffers

def recover_octree(octree_path, node, level):
    if level <= 16:
        position_buffer = node.pointcloud.points
        color_buffer = node.pointcloud.colors
        name = node.name
        output_path = os.path.join(octree_path, f"level_{level}_{name}.ply")
        vertices = np.array(
            [(position[0], position[1], position[2], color[0], color[1], color[2]) for position, color in zip(position_buffer, color_buffer)],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el], text=True).write(output_path)
        if hasattr(node, 'children'):
            for child in node.children:
                if child is not None:
                    recover_octree(octree_path, child, level + 1)  

def readoctreeColmapInfo(path, images, depths, eval, llffhold=8):
    """
    读取八叉树LOD Colmap场景信息的函数
    
    功能描述：从Colmap数据生成多层级LOD八叉树点云，支持细节层级的动态加载
    
    参数：
    @param path: Colmap数据集的根路径
    @param images: 图像文件夹名称，None时使用默认的"images"
    @param depths: 深度图文件夹名称，空字符串表示无深度数据
    @param eval: 是否为评估模式，True时会分割训练集和测试集
    @param llffhold: LLFF风格的数据分割参数，每隔llffhold张图像作为测试集
    
    @return: SceneInfo对象，包含多层级LOD点云、相机信息和归一化参数
    """
    # ==================== 相机参数读取部分（与标准Colmap相同）====================
    # 尝试读取二进制格式的Colmap输出文件
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        # 回退到文本格式文件
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 确定图像文件夹路径
    images_dir = "images" if images == None else images
    
    # 读取相机信息（包括图像和可选的深度数据）
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, images_dir),
                                           depths_folder=os.path.join(path, depths) if depths != "" else None)
    # 按图像名称排序
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 数据集分割：训练集和测试集
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # 计算NeRF++风格的场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ==================== 八叉树LOD点云处理部分 ====================
    # 定义各种文件路径
    octree_path = os.path.join(path, "octree")                    # 八叉树输出目录
    las_path = os.path.join(path, "octree/0/points3D.las")        # LAS格式点云文件
    ply_path = os.path.join(path, "sparse/0/points3D.ply")        # PLY格式点云文件
    bin_path = os.path.join(path, "sparse/0/points3D.bin")        # 二进制格式点云文件
    txt_path = os.path.join(path, "sparse/0/points3D.txt")        # 文本格式点云文件

    pcd = None
    
    # 检查LAS文件是否存在，如果不存在则进行转换流程
    if not os.path.exists(las_path):
        print("[ Dataloader ] Converting point3d.bin to LOD PCS, will happen only the first time you open the scene.")
        
        # 首先确保有PLY格式的点云文件
        if not os.path.exists(ply_path):
            # PLY文件不存在，从Colmap原始格式转换
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)    # 尝试读取二进制格式
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)      # 回退到文本格式
            storePly(ply_path, xyz, rgb)                        # 保存为PLY格式
        else:
            # PLY文件已存在，直接加载
            print("[ Dataloader ] Found .ply point cloud, skipping conversion.")
            pcd = fetchPly(ply_path)
            xyz, rgb = pcd.points, pcd.colors

        # 转换PLY到LAS格式（LAS是激光雷达数据的标准格式，支持更好的八叉树处理）
        print("[ Dataloader ] Converting points to LAS format.")
        mkdir_p(os.path.dirname(las_path))                      # 创建目录
        storeLas(las_path, xyz, rgb)                            # 保存为LAS格式

        # 使用PotreeConverter将LAS转换为八叉树结构
        print("[ Dataloader ] Converting LAS to octree for level-of-detail pointclouds.")
        # 调用外部PotreeConverter工具生成八叉树
        command = [os.path.join(os.getcwd(), "PotreeConverter/bin/Release/Converter.exe"), las_path, "-o", octree_path, "--overwrite"]
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[ Dataloader ] LOD pointclouds generated.")
    else:
        # LAS文件存在，检查是否已经生成了八叉树
        if not os.path.exists(os.path.join(octree_path, "metadata.json")):
            # 八叉树元数据不存在，重新生成八叉树
            print("[ Dataloader ] Converting LAS to octree for level-of-detail pointclouds.")
            command = [os.path.join(os.getcwd(), "PotreeConverter/bin/Release/Converter.exe"), las_path, "-o", octree_path, "--overwrite"]
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("[ Dataloader ] LOD pointclouds generated.")
        else:
            # 八叉树已存在，跳过转换
            print("[ Dataloader ] Found octree, skipping conversion.")

    # ==================== 八叉树数据加载和层级分析 ====================
    # 初始化八叉树场景
    octreeGaussian = loadOctree(octree_path)                    # 加载八叉树结构
    octreeGaussianLoader = octreeGaussian.loader                # 获取八叉树加载器
    max_level = 0                                               # 最大层级深度
    
    # 使用广度优先搜索遍历八叉树，计算最大层级深度
    q = queue.Queue()
    q.put({"node": octreeGaussian.root, "level": 0})           # 从根节点开始
    
    while q.qsize() > 0:
        element = q.get()
        node = element["node"]                                  # 当前节点
        max_level = level = element["level"]                    # 当前层级
        
        # 限制最大层级为16（避免过深的递归）
        if level < 16:
            octreeGaussianLoader.load(node)                     # 加载当前节点的点云数据

            # 遍历当前节点的8个子节点（八叉树每个节点有8个子节点）
            for cid in range(8):
                child = node.children[cid]
                if child is not None:
                    # 将子节点加入队列，层级+1
                    q.put({"node": child, "level": level + 1})

    # ==================== 注释掉的代码：合并所有层级的点云 ====================
    # 以下代码被注释掉了，原本用于将所有层级的点云合并为单个文件
    # concat all the position and color buffers into single ply
    # position_buffers, color_buffers = collect_position_buffers(octreeGaussian.root, 0, max_level)
    # all_positions = np.concatenate(position_buffers, axis=0)
    # all_color = np.concatenate(color_buffers, axis=0)
    # vertices = np.array(
    #     [(position[0], position[1], position[2], color[0], color[1], color[2]) for position, color in zip(all_positions, all_color)],
    #     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    # )
    # el = PlyElement.describe(vertices, 'vertex')
    # PlyData([el], text=True).write(os.path.join(octree_path, "pcd_octree.ply"))

    # 注释掉的代码：单独保存每个层级
    # do not concat, save them individually
    # recover_octree(octree_path, octreeGaussian.root, 0)

    # ==================== 为每个LOD层级创建点云数据 ====================
    pcds = []
    
    # 遍历所有LOD层级（从0到最大层级）
    for level in range(0, max_level + 1):
        # 收集当前层级及以下所有层级的点云数据
        # 层级越低，包含的点越多（累积效果）
        position_buffers, color_buffers = collect_position_buffers(octreeGaussian.root, 0, level)
        
        # 合并当前层级的位置和颜色数据
        positions = np.concatenate(position_buffers, axis=0)    # 合并位置坐标
        colors = np.concatenate(color_buffers, axis=0)          # 合并颜色信息
        normals = np.zeros_like(positions)                      # 创建零法向量（占位符）
        
        # 创建基础点云对象并添加到列表
        # 颜色值从[0,255]范围归一化到[0,1]
        pcds.append(BasicPointCloud(positions, colors[:,:3] / 255.0, normals))
    
    # ==================== 创建场景信息对象 ====================
    scene_info = SceneInfo(
        point_cloud=pcds,                      # 多层级LOD点云数据列表
        train_cameras=train_cam_infos,         # 训练相机信息
        test_cameras=test_cam_infos,           # 测试相机信息
        nerf_normalization=nerf_normalization, # 场景归一化参数
        ply_path=ply_path,                     # 原始PLY文件路径
        max_level=max_level                    # 最大LOD层级数
    )
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Octree" : readoctreeColmapInfo
}
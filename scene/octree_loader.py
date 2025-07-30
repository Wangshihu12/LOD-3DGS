# ref: potree\src\modules\loader\2.0\octreeGaussian.js
# create by Penghao Wang

import os
import numpy as np
import json

from utils.graphics_utils import OctreeGaussianNode, Vector3, BoundingBox, OctreeGaussian
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

octreeConst = {
    "pointBudget": 1 * 1000 * 1000,
    "framenumber" : 0,
    "numNodesLoading" : 0,
    "maxNodesLoading" : 4
}

PointAttributeTypesTmp = {
    "DATA_TYPE_DOUBLE": {"ordinal": 0, "name": "double", "size": 8},
    "DATA_TYPE_FLOAT":  {"ordinal": 1, "name": "float",  "size": 4},
    "DATA_TYPE_INT8":   {"ordinal": 2, "name": "int8",   "size": 1},
    "DATA_TYPE_UINT8":  {"ordinal": 3, "name": "uint8",  "size": 1},
    "DATA_TYPE_INT16":  {"ordinal": 4, "name": "int16",  "size": 2},
    "DATA_TYPE_UINT16": {"ordinal": 5, "name": "uint16", "size": 2},
    "DATA_TYPE_INT32":  {"ordinal": 6, "name": "int32",  "size": 4},
    "DATA_TYPE_UINT32": {"ordinal": 7, "name": "uint32", "size": 4},
    "DATA_TYPE_INT64":  {"ordinal": 8, "name": "int64",  "size": 8},
    "DATA_TYPE_UINT64": {"ordinal": 9, "name": "uint64", "size": 8}
}

PointAttributeTypes = PointAttributeTypesTmp.copy()

i = 0
for obj in PointAttributeTypesTmp:
    PointAttributeTypes[str(i)] = PointAttributeTypesTmp[obj]
    i += 1

# print(PointAttributeTypes)

class PointAttribute:
    def __init__(self, name, type, numElements):
        self.name = name
        self.type = type
        self.numElements = numElements
        self.byteSize = self.numElements * self.type['size']
        self.description = ""
        self.range = [float('inf'), float('-inf')]

# Defining the static attributes for the PointAttribute class
PointAttribute.POSITION_CARTESIAN = PointAttribute("POSITION_CARTESIAN", PointAttributeTypes["DATA_TYPE_FLOAT"], 3)
PointAttribute.RGBA_PACKED = PointAttribute("COLOR_PACKED", PointAttributeTypes["DATA_TYPE_INT8"], 4)
PointAttribute.COLOR_PACKED = PointAttribute.RGBA_PACKED
PointAttribute.RGB_PACKED = PointAttribute("COLOR_PACKED", PointAttributeTypes["DATA_TYPE_INT8"], 3)
PointAttribute.NORMAL_FLOATS = PointAttribute("NORMAL_FLOATS", PointAttributeTypes["DATA_TYPE_FLOAT"], 3)
PointAttribute.INTENSITY = PointAttribute("INTENSITY", PointAttributeTypes["DATA_TYPE_UINT16"], 1)
PointAttribute.CLASSIFICATION = PointAttribute("CLASSIFICATION", PointAttributeTypes["DATA_TYPE_UINT8"], 1)
PointAttribute.NORMAL_SPHEREMAPPED = PointAttribute("NORMAL_SPHEREMAPPED", PointAttributeTypes["DATA_TYPE_UINT8"], 2)
PointAttribute.NORMAL_OCT16 = PointAttribute("NORMAL_OCT16", PointAttributeTypes["DATA_TYPE_UINT8"], 2)
PointAttribute.NORMAL = PointAttribute("NORMAL", PointAttributeTypes["DATA_TYPE_FLOAT"], 3)
PointAttribute.RETURN_NUMBER = PointAttribute("RETURN_NUMBER", PointAttributeTypes["DATA_TYPE_UINT8"], 1)
PointAttribute.NUMBER_OF_RETURNS = PointAttribute("NUMBER_OF_RETURNS", PointAttributeTypes["DATA_TYPE_UINT8"], 1)
PointAttribute.SOURCE_ID = PointAttribute("SOURCE_ID", PointAttributeTypes["DATA_TYPE_UINT16"], 1)
PointAttribute.INDICES = PointAttribute("INDICES", PointAttributeTypes["DATA_TYPE_UINT32"], 1)
PointAttribute.SPACING = PointAttribute("SPACING", PointAttributeTypes["DATA_TYPE_FLOAT"], 1)
PointAttribute.GPS_TIME = PointAttribute("GPS_TIME", PointAttributeTypes["DATA_TYPE_DOUBLE"], 1)

class PointAttributes:
    def __init__(self, pointAttributes=None):
        self.attributes = []
        self.byteSize = 0
        self.size = 0
        self.vectors = []

        if pointAttributes is not None:
            for pointAttributeName in pointAttributes:
                pointAttribute = getattr(PointAttribute, pointAttributeName, None)
                if pointAttribute:
                    self.attributes.append(pointAttribute)
                    self.byteSize += pointAttribute.byteSize
                    self.size += 1

    def add(self, pointAttribute):
        self.attributes.append(pointAttribute)
        self.byteSize += pointAttribute.byteSize
        self.size += 1

    def addVector(self, vector):
        self.vectors.append(vector)

typename_typeattribute_map = {
    "double": PointAttributeTypes["DATA_TYPE_DOUBLE"],
    "float": PointAttributeTypes["DATA_TYPE_FLOAT"],
    "int8": PointAttributeTypes["DATA_TYPE_INT8"],
    "uint8": PointAttributeTypes["DATA_TYPE_UINT8"],
    "int16": PointAttributeTypes["DATA_TYPE_INT16"],
    "uint16": PointAttributeTypes["DATA_TYPE_UINT16"],
    "int32": PointAttributeTypes["DATA_TYPE_INT32"],
    "uint32": PointAttributeTypes["DATA_TYPE_UINT32"],
    "int64": PointAttributeTypes["DATA_TYPE_INT64"],
    "uint64": PointAttributeTypes["DATA_TYPE_UINT64"],
}

tmpVec3 = Vector3()

def createChildAABB(aabb: BoundingBox, index: int) -> BoundingBox:
    minPoint = Vector3(aabb.min.x, aabb.min.y, aabb.min.z)
    maxPoint = Vector3(aabb.max.x, aabb.max.y, aabb.max.z)
    size = tmpVec3.subVectors(maxPoint, minPoint)

    if (index & 0b0001) > 0:
        minPoint.z += size.z / 2
    else:
        maxPoint.z -= size.z / 2

    if (index & 0b0010) > 0:
        minPoint.y += size.y / 2
    else:
        maxPoint.y -= size.y / 2

    if (index & 0b0100) > 0:
        minPoint.x += size.x / 2
    else:
        maxPoint.x -= size.x / 2

    return BoundingBox(min_point=minPoint, max_point=maxPoint)

def loadOctree(path):
    if not os.path.exists(path):
        return None
    
    if "metadata.json" not in os.listdir(path):
        assert False, "[ Error ] Octree path dir does not contain metadata.json in loadOctree method"

    loadworker = octreeLoader()
    octree = loadworker.load(path)
    return octree

def toIndex(x, y, z, sizeX, sizeY, sizeZ):
    gridSize = 32
    dx = gridSize * x / sizeX
    dy = gridSize * y / sizeY
    dz = gridSize * z / sizeZ

    # print(dx, gridSize)

    ix = min(int(dx), gridSize - 1)
    iy = min(int(dy), gridSize - 1)
    iz = min(int(dz), gridSize - 1)

    index = ix + iy * gridSize + iz * gridSize * gridSize
    return index

class nodeLoader():
    """
    八叉树节点加载器类
    
    功能描述：负责从二进制文件中加载八叉树节点的层级结构和点云数据
    """
    
    def __init__(self, path: str) -> None:
        """
        初始化节点加载器
        
        参数：
        @param path: 八叉树数据文件的根路径
        """
        self.path = path          # 八叉树数据目录路径
        self.metadata = None      # 元数据信息
        self.attributes = None    # 点云属性定义
        self.scale = None         # 坐标缩放因子
        self.offset = None        # 坐标偏移量

    def loadHierarchy(self, node: OctreeGaussianNode) -> None:
        """
        加载八叉树节点的层级结构数据
        
        功能描述：从hierarchy.bin文件中读取指定节点的层级结构信息
        
        参数：
        @param node: 要加载层级数据的八叉树节点
        """
        # 获取层级数据在二进制文件中的位置信息
        hierarchyByteOffset = node.hierarchyByteOffset    # 层级数据的字节偏移
        hierarchyByteSize = node.hierarchyByteSize        # 层级数据的字节大小
        first = hierarchyByteOffset                       # 起始字节位置
        last = first + hierarchyByteSize - 1              # 结束字节位置
        
        # 从hierarchy.bin文件加载指定范围的字节数据
        hierarchyPath = os.path.join(self.path, "hierarchy.bin")
        with open(hierarchyPath, "rb") as f:
            # 定位到起始字节位置
            f.seek(first)
            # 读取指定大小的二进制数据
            buffer = f.read(last - first + 1)
        f.close()
        
        # 解析读取的二进制层级数据
        self.parseHierarchy(node, buffer)

    def parseHierarchy(self, node: OctreeGaussianNode, buffer):
        """
        解析八叉树层级结构的二进制数据
        
        功能描述：将二进制缓冲区中的层级数据解析为八叉树节点结构
        
        参数：
        @param node: 当前处理的八叉树节点
        @param buffer: 包含层级数据的二进制缓冲区
        """
        # 每个节点在二进制文件中占用22字节
        bytesPerNode = 22
        # 计算缓冲区中包含的节点数量
        numNodes = int(len(buffer) / bytesPerNode)

        # 获取八叉树对象引用
        octree = node.octreeGaussian
        # 创建节点数组，用于存储所有解析的节点
        nodes = [None for i in range(numNodes)]
        nodes[0] = node          # 第一个节点是当前传入的节点
        nodePos = 1              # 下一个要填充的节点位置

        # 遍历缓冲区中的每个节点数据
        for i in range(numNodes):
            # 计算当前节点数据在缓冲区中的起始位置
            start = i * bytesPerNode
            
            # 解析节点的各项属性（按字节顺序）：
            # uint8: 节点类型
            type = buffer[start]
            # uint8: 子节点掩码（8位，每位表示一个子节点是否存在）
            childMask = buffer[start + 1]
            # uint32: 该节点包含的点数量
            numPoints = int.from_bytes(buffer[start + 2:start + 6], byteorder='little', signed=False)
            # int64: 数据在文件中的字节偏移
            byteOffset = int.from_bytes(buffer[start + 6:start + 14], byteorder='little', signed=True)
            # int64: 数据的字节大小
            byteSize = int.from_bytes(buffer[start + 14:start + 22], byteorder='little', signed=True)

            # 调试信息（已注释）
            # print(f"[ Info ] type: {type}, childMask: {childMask}, numPoints: {numPoints}, byteOffset: {byteOffset}, byteSize: {byteSize}")

            # 根据节点类型设置不同的属性
            if nodes[i].nodeType == 2:
                # 类型2：叶子节点或数据节点
                nodes[i].byteOffset = byteOffset
                nodes[i].byteSize = byteSize
                nodes[i].numGaussians = numPoints
            elif type == 2:
                # 新的类型2节点：包含层级信息
                nodes[i].hierarchyByteOffset = byteOffset
                nodes[i].hierarchyByteSize = byteSize
                nodes[i].numGaussians = numPoints
            else:
                # 其他类型节点：普通内部节点
                nodes[i].byteOffset = byteOffset
                nodes[i].byteSize = byteSize
                nodes[i].numGaussians = numPoints

            # 如果字节大小为0，说明该节点没有高斯点数据
            if nodes[i].byteSize == 0:
                nodes[i].numGaussians = 0

            # 设置节点类型
            nodes[i].nodeType = type

            # 如果是类型2节点，跳过子节点创建（可能是叶子节点）
            if nodes[i].nodeType == 2:
                continue

            # 为当前节点创建子节点（八叉树每个节点最多有8个子节点）
            for childIndex in range(8):
                # 检查子节点掩码，确定第childIndex个子节点是否存在
                childExists = ((1 << childIndex) & childMask) != 0
                if not childExists:
                    continue

                # 生成子节点名称（在父节点名称后添加索引）
                childName = nodes[i].name + str(childIndex)
                # 计算子节点的边界框（将父节点边界框分割为8个子区域）
                childAABB = createChildAABB(nodes[i].boundingbox, childIndex)
                # 创建新的子节点对象
                child = OctreeGaussianNode(childName, octree, childAABB)
                child.name = childName
                child.spacing = nodes[i].spacing / 2    # 子节点间距是父节点的一半
                child.level = nodes[i].level + 1        # 子节点层级比父节点高1
                child.parent = nodes[i]                 # 设置父节点引用

                # 将子节点添加到父节点的子节点数组中
                nodes[i].children[childIndex] = child
                # 将子节点添加到待处理的节点列表中
                nodes[nodePos] = child
                nodePos += 1

    def load(self, node: OctreeGaussianNode) -> None:
        """
        加载八叉树节点的具体点云数据
        
        功能描述：从octree.bin文件中读取节点的点云数据（位置、颜色等）
        
        参数：
        @param node: 要加载数据的八叉树节点
        """

        # 调试信息（已注释）
        # print(node)

        # 检查节点是否已经加载或正在加载，避免重复加载
        if (node.loaded or node.loading):
            return
        
        # 标记节点正在加载，并更新全局加载计数器
        node.loading = True
        octreeConst["numNodesLoading"] += 1
        
        # 如果是类型2节点，需要先加载其层级结构
        if node.nodeType == 2:
            self.loadHierarchy(node=node)

        # 获取节点数据在octree.bin文件中的位置信息
        byteOffset = node.byteOffset    # 数据起始字节偏移
        byteSize = node.byteSize        # 数据字节大小

        # 构建octree.bin文件路径
        octreePath = os.path.join(self.path, "octree.bin")

        # 计算要读取的字节范围
        first = byteOffset
        last = first + byteSize - 1

        # 检查数据大小有效性
        if byteSize == 0:
            assert False, "[ Error ] byteSize is 0 in nodeLoader.load method"
        else:
            # 从octree.bin文件读取指定范围的数据
            with open(octreePath, "rb") as f:
                f.seek(first)
                buffer = f.read(last - first + 1)
            f.close()

        # 初始化属性缓冲区和偏移量
        attributeBuffers = {}    # 存储各种属性的数据缓冲区
        attributeOffset = 0      # 当前属性在点数据中的字节偏移

        # 计算每个点占用的总字节数
        bytesPerPoint = 0
        for pointAttribute in node.octreeGaussian.pointAttributes.attributes:
            bytesPerPoint += pointAttribute.byteSize

        # 获取坐标转换参数
        scale = node.octreeGaussian.scale          # 缩放因子
        offset = node.octreeGaussian.loader.offset # 偏移量

        # 调试信息（已注释）
        # print(node.octreeGaussian.loader.offset)
        # print(node.octreeGaussian.offset)

        # 遍历所有点云属性，解析相应的数据
        for pointAttribute in node.octreeGaussian.pointAttributes.attributes:
            if pointAttribute.name in ["POSITION_CARTESIAN", "position"]:
                # 处理位置属性（3D坐标）
                buff = np.zeros(node.numGaussians * 3, dtype=np.float32)
                positions = buff
                
                # 逐个解析每个点的坐标
                for j in range(node.numGaussians):
                    pointOffset = j * bytesPerPoint    # 当前点在缓冲区中的起始位置

                    # 从二进制数据中解析xyz坐标（小端序32位有符号整数）
                    # 解析后应用缩放和偏移变换，保持与Colmap坐标系的一致性
                    x = (int.from_bytes(buffer[pointOffset + attributeOffset + 0:pointOffset + attributeOffset + 4], byteorder="little", signed=True) * scale[0]) + offset[0]
                    y = (int.from_bytes(buffer[pointOffset + attributeOffset + 4:pointOffset + attributeOffset + 8], byteorder="little", signed=True) * scale[1]) + offset[1]
                    z = (int.from_bytes(buffer[pointOffset + attributeOffset + 8:pointOffset + attributeOffset + 12], byteorder="little", signed=True) * scale[2]) + offset[2]
                    
                    # 将坐标存储到缓冲区中
                    positions[3 * j + 0] = x
                    positions[3 * j + 1] = y
                    positions[3 * j + 2] = z

                # 将位置数据添加到属性缓冲区
                attributeBuffers[pointAttribute.name] = {"buffer": buff, "attribute": pointAttribute}
                
            elif pointAttribute.name in ["RGBA", "rgba"]:
                # 处理颜色属性（RGBA）
                buff = np.zeros(node.numGaussians * 4, dtype = np.uint8)
                colors = buff

                # 逐个解析每个点的颜色
                for j in range(node.numGaussians):
                    pointOffset = j * bytesPerPoint
                    
                    # 从二进制数据中解析RGB颜色值（16位无符号整数）
                    r = np.frombuffer(buffer[pointOffset + attributeOffset + 0:pointOffset + attributeOffset + 2], dtype=np.uint16)[0]
                    g = np.frombuffer(buffer[pointOffset + attributeOffset + 2:pointOffset + attributeOffset + 4], dtype=np.uint16)[0]
                    b = np.frombuffer(buffer[pointOffset + attributeOffset + 4:pointOffset + attributeOffset + 6], dtype=np.uint16)[0]

                    # 将16位颜色值转换为8位（处理大于255的情况）
                    colors[4 * j + 0] = r / 256 if r > 255 else r
                    colors[4 * j + 1] = g / 256 if g > 255 else g
                    colors[4 * j + 2] = b / 256 if b > 255 else b
                    # Alpha通道默认为0（在初始化时已设置）
                    
                # 将颜色数据添加到属性缓冲区
                attributeBuffers[pointAttribute.name] = {"buffer": buff, "attribute": pointAttribute}

            else:
                # 其他属性暂时不处理
                pass

            # 更新属性偏移量，指向下一个属性
            attributeOffset += pointAttribute.byteSize

        # 将位置数据重新组织为Nx3的数组格式
        node_position = np.array(attributeBuffers["position"]["buffer"], dtype=np.float32).reshape(-1, 3)
        
        try:
            # 尝试将颜色数据重新组织为Nx4的数组格式
            node_colors = np.array(attributeBuffers["rgba"]["buffer"], dtype=np.uint8).reshape(-1, 4)
        except:
            # 如果颜色数据不存在，创建零数组作为占位符
            node_colors = np.zeros(len(attributeBuffers["position"]["buffer"]))

        # 创建基础点云对象
        pcd = BasicPointCloud(node_position, node_colors, None)  # None表示没有法向量数据

        # 将点云数据分配给节点
        node.pointcloud = pcd

        # 标记节点加载完成，并更新加载状态
        node.loaded = True
        node.loading = False
        octreeConst["numNodesLoading"] -= 1

class octreeLoader():
    """
    八叉树加载器类
    
    功能描述：负责加载和解析八叉树格式的点云数据，构建LOD层级结构
    """

    def __init__(self) -> None:
        """
        初始化八叉树加载器
        """
        self.metadata = None  # 八叉树元数据，包含结构信息和属性定义

    def load(self, path_dir):
        """
        加载八叉树数据的主函数
        
        功能描述：从指定目录加载八叉树元数据，创建八叉树结构和根节点
        
        参数：
        @param path_dir: 八叉树数据目录路径
        
        @return: OctreeGaussian对象，包含完整的八叉树结构
        """

        # 构建元数据文件路径
        path = os.path.join(path_dir, "metadata.json")

        # 检查元数据文件是否存在
        if not os.path.exists(path):
            assert False, "[ Error ] Path does not exist in disk in octreeLoader.load method"
        
        # 读取并解析JSON格式的元数据文件
        with open(path, 'r') as file:
            self.metadata = json.load(file)  # 加载八叉树的结构信息、边界框、属性等
        file.close()

        # 解析点云属性定义（位置、颜色、法向量等）
        attributes = self.parseAttributes(self.metadata["attributes"])

        # 创建节点加载器，负责加载具体的点云数据
        loader = nodeLoader(path_dir)
        loader.metadata = self.metadata          # 传递元数据
        loader.attributes = attributes            # 传递属性定义
        loader.scale = self.metadata["scale"]     # 点云缩放因子
        loader.offset = self.metadata["offset"]   # 点云偏移量

        # 创建八叉树高斯对象
        octree = OctreeGaussian()
        octree.spacing = self.metadata["spacing"]  # 八叉树节点间距
        octree.scale = self.metadata["scale"]      # 缩放因子

        # 解析边界框信息
        meta_min = self.metadata["boundingBox"]["min"]  # 边界框最小值 [x_min, y_min, z_min]
        meta_max = self.metadata["boundingBox"]["max"]  # 边界框最大值 [x_max, y_max, z_max]

        # 创建3D向量表示边界框的最小和最大坐标
        min = Vector3(meta_min[0], meta_min[1], meta_min[2])
        max = Vector3(meta_max[0], meta_max[1], meta_max[2])
        boundingBox = BoundingBox(min, max)

        # 创建偏移向量，用于坐标归一化
        offset = Vector3(meta_min[0], meta_min[1], meta_min[2])

        # 对边界框应用偏移，将坐标系原点移动到边界框最小值
        boundingBox.min -= offset
        boundingBox.max -= offset

        # 设置八叉树的各项属性
        octree.projection = self.metadata["projection"]                             # 投影信息
        octree.boundingBox = boundingBox                                           # 归一化后的边界框
        octree.offset = offset                                                     # 坐标偏移量
        octree.pointAttributes = self.parseAttributes(self.metadata["attributes"]) # 点云属性
        octree.loader = loader                                                     # 节点加载器

        # 创建八叉树根节点
        root = OctreeGaussianNode("r", octree, boundingBox)  # "r"表示根节点
        root.level = 0                                       # 根节点层级为0
        root.nodeType = 2                                    # 节点类型（2可能表示根节点类型）
        root.hierarchyByteOffset = 0                         # 层级数据在文件中的字节偏移
        root.hierarchyByteSize = self.metadata["hierarchy"]["firstChunkSize"]  # 第一个数据块的大小
        root.spacing = octree.spacing                        # 节点间距
        root.byteOffset = 0                                  # 数据字节偏移

        # 将根节点分配给八叉树
        octree.root = root

        # 使用加载器加载根节点的数据（这会触发递归加载子节点）
        loader.load(root)

        return octree

    def parseAttributes(self, jsonAttributes: list) -> None:
        """
        解析点云属性定义的函数
        
        功能描述：将JSON格式的属性定义转换为内部的PointAttributes对象
        
        参数：
        @param jsonAttributes: JSON格式的属性列表，每个属性包含名称、类型、大小等信息
        
        @return: PointAttributes对象，包含所有解析后的属性定义
        """
        # 创建点云属性容器
        attributes = PointAttributes()
        
        # 属性名称替换映射（处理命名不一致问题）
        replacements = {
            "rgb": "rgba"  # 将"rgb"属性名替换为"rgba"
        }
        
        # 遍历所有属性定义
        for jsonAttribute in jsonAttributes:
            # 提取属性的各项参数
            name = jsonAttribute["name"]                    # 属性名称（如"position", "rgb"等）
            description = jsonAttribute["description"]      # 属性描述
            size = jsonAttribute["size"]                    # 属性总字节大小
            numElements = jsonAttribute["numElements"]      # 元素数量（如RGB有3个元素）
            elementSize = jsonAttribute["elementSize"]      # 单个元素字节大小
            type = jsonAttribute["type"]                    # 数据类型（如"float", "uint8"等）
            min = jsonAttribute["min"]                      # 属性值最小值
            max = jsonAttribute["max"]                      # 属性值最大值

            # 将字符串类型转换为内部类型枚举
            type = typename_typeattribute_map[type]
            
            # 应用属性名称替换（如果需要）
            potreeAttributeName = replacements[name] if name in replacements else name
            
            # 创建点云属性对象
            attribute = PointAttribute(potreeAttributeName, type, numElements)

            # 设置属性值范围
            if numElements == 1:
                # 标量属性：直接使用min[0]和max[0]
                attribute.range = [min[0], max[0]]
            else:
                # 向量属性：使用完整的min和max数组
                attribute.range = [min, max]

            # 特殊处理GPS时间属性（避免范围为0的情况）
            if name == "gps-time":
                if attribute.range[0] == attribute.range[1]:
                    attribute.range[1] += 1  # 确保范围不为0

            # 将属性添加到属性容器中
            attributes.add(attribute)

        # 检查是否包含法向量属性（NormalX, NormalY, NormalZ）
        if any(attr.name == "NormalX" for attr in attributes.attributes) and \
            any(attr.name == "NormalY" for attr in attributes.attributes) and \
            any(attr.name == "NormalZ" for attr in attributes.attributes):
            
            # 如果包含完整的法向量分量，创建法向量向量属性
            vector = {
                "name": "NORMAL",                                    # 向量名称
                "attributes": ["NormalX", "NormalY", "NormalZ"]     # 组成向量的属性列表
            }
            attributes.addVector(vector)  # 添加向量属性

        return attributes
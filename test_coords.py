import numpy as np
import open3d as o3d
import os
from PIL import Image, ImageDraw
import json
import pickle
import cv2

from config import cfg

class_type_to_number = {
        'ape': '001',
        'benchvise': '002',
        'cam': '003',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010',
        'iron': '011',
        'lamp': '012',
        'phone': '013'
    }

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

linemod_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue',
                     'holepuncher', 'iron', 'lamp', 'phone']


def project_K(pts_3d, RT, K):
    pts_2d = np.matmul(pts_3d, RT[:, :3].T) + RT[:, 3:].T
    pts_2d = np.matmul(pts_2d, K.T)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def plot_box(imgpath, proj_gt, proj_pr, show=False):
    """
    proj_gt: [n, 2]
    proj_pr: [n, 2]
    """
    img = Image.open(imgpath).convert('RGB')
    draw = ImageDraw.Draw(img)
    color_gt = (0, 255, 0)
    color_pr = (0, 0, 255)

    # color_pr = (255, 255, 255)  # 不同框变换颜色 白色
    # color_pr = (255, 0, 0)  # 不同框变换颜色 红色
    # color_pr = (0, 255, 0)  # 不同框变换颜色 绿
    # color_pr = (0, 0, 255)  # 不同框变换颜色 蓝
    # color_pr = (255, 255, 0)  # 不同框变换颜色 黄色
    # color_pr = (255, 0, 255)  # 不同框变换颜色 紫色
    # color_pr = (0, 255, 255)  # 不同框变换颜色 青色
    # color_pr = (255, 20, 147)  # 不同框变换颜色 深粉色
    # color_pr = (99, 184, 255)  # 不同框变换颜色 淡蓝色

    # 绘制真实边界框
    draw.line(((proj_gt[0][0], proj_gt[0][1]), (proj_gt[1][0], proj_gt[1][1]),
               (proj_gt[3][0], proj_gt[3][1]), (proj_gt[2][0], proj_gt[2][1]),
               (proj_gt[0][0], proj_gt[0][1])), fill=color_gt, width=2)
    draw.line(((proj_gt[4][0], proj_gt[4][1]), (proj_gt[5][0], proj_gt[5][1]),
               (proj_gt[7][0], proj_gt[7][1]), (proj_gt[6][0], proj_gt[6][1]),
               (proj_gt[4][0], proj_gt[4][1])), fill=color_gt, width=2)
    draw.line(((proj_gt[6][0], proj_gt[6][1]), (proj_gt[7][0], proj_gt[7][1]),
               (proj_gt[3][0], proj_gt[3][1]), (proj_gt[2][0], proj_gt[2][1]),
               (proj_gt[6][0], proj_gt[6][1])), fill=color_gt, width=2)
    draw.line(((proj_gt[0][0], proj_gt[0][1]), (proj_gt[1][0], proj_gt[1][1]),
               (proj_gt[5][0], proj_gt[5][1]), (proj_gt[4][0], proj_gt[4][1]),
               (proj_gt[0][0], proj_gt[0][1])), fill=color_gt, width=2)

    # 绘制预测边框
    draw.line(((proj_pr[0][0], proj_pr[0][1]), (proj_pr[1][0], proj_pr[1][1]),
               (proj_pr[3][0], proj_pr[3][1]), (proj_pr[2][0], proj_pr[2][1]),
               (proj_pr[0][0], proj_pr[0][1])), fill=color_pr, width=2)
    draw.line(((proj_pr[4][0], proj_pr[4][1]), (proj_pr[5][0], proj_pr[5][1]),
               (proj_pr[7][0], proj_pr[7][1]), (proj_pr[6][0], proj_pr[6][1]),
               (proj_pr[4][0], proj_pr[4][1])), fill=color_pr, width=2)
    draw.line(((proj_pr[6][0], proj_pr[6][1]), (proj_pr[7][0], proj_pr[7][1]),
               (proj_pr[3][0], proj_pr[3][1]), (proj_pr[2][0], proj_pr[2][1]),
               (proj_pr[6][0], proj_pr[6][1])), fill=color_pr, width=2)
    draw.line(((proj_pr[0][0], proj_pr[0][1]), (proj_pr[1][0], proj_pr[1][1]),
               (proj_pr[5][0], proj_pr[5][1]), (proj_pr[4][0], proj_pr[4][1]),
               (proj_pr[0][0], proj_pr[0][1])), fill=color_pr, width=2)

    if show:
        img.show()

    return img


def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_EPNP):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    # _, R_exp, t = cv2.solvePnP(points_3d.reshape((-1, 1, 3)),
    #                            points_2d.reshape((-1, 1, 2)),
    #                            camera_matrix,
    #                            dist_coeffs,
    #                            flags=method)
    # , None, None, False, cv2.SOLVEPNP_UPNP)
    _, R_exp, t, inliers = cv2.solvePnPRansac(points_3d.reshape((-1, 1, 3)),
                                              points_2d.reshape((-1, 1, 2)),
                                              camera_matrix,
                                              dist_coeffs,
                                              # iterationsCount=1000,
                                              reprojectionError=1.5,
                                              flags=cv2.SOLVEPNP_EPNP
                                              # flags=cv2.SOLVEPNP_ITERATIVE
                                              )
    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)


if __name__ == "__main__":

    for idx in range(len(os.listdir(os.path.join(cfg.OUTPUT_DIR, 'rgb')))):
        img_path = os.path.join(cfg.OUTPUT_DIR, 'rgb/{:06}.jpg'.format(idx))

        mask_path = os.path.join(cfg.OUTPUT_DIR, 'mask/{:06}.png'.format(idx))
        mask = Image.open(mask_path).convert('I')
        mask = np.array(mask, np.uint8)

        x_coords = np.load(os.path.join(cfg.OUTPUT_DIR, 'coords/x/{:06}.npy'.format(idx))).astype('float32')
        y_coords = np.load(os.path.join(cfg.OUTPUT_DIR, 'coords/y/{:06}.npy'.format(idx))).astype('float32')
        z_coords = np.load(os.path.join(cfg.OUTPUT_DIR, 'coords/z/{:06}.npy'.format(idx))).astype('float32')

        coords = np.stack((x_coords, y_coords, z_coords), axis=2)  # [h, w, 3]

        meta_info_path = os.path.join(cfg.OUTPUT_DIR, 'meta_info/{:06}.pkl'.format(idx))

        begins, poses, bboxs = read_pickle(meta_info_path)

        xmap = np.array([[i for i in range(mask.shape[1])] for j in range(mask.shape[0])])  # [h, w]
        ymap = np.array([[j for i in range(mask.shape[1])] for j in range(mask.shape[0])])  # [h, w]

        xmap = xmap.reshape(xmap.shape[0], xmap.shape[1], 1)
        ymap = ymap.reshape(ymap.shape[0], ymap.shape[1], 1)

        xymap = np.concatenate((xmap, ymap), axis=2)  # [h, w, 2]

        for cls in linemod_cls_names:
            # cls = 'cat'
            print('可视化图像： {} 中的 {}'.format(idx, cls))
            cls_num = int(class_type_to_number[cls])
            ply_path = os.path.join(cfg.LINEMOD, '{}/{}.ply'.format(cls, cls))
            ply_model = o3d.io.read_triangle_mesh(ply_path)  # 加载mesh
            obj_pcd = ply_model.sample_points_uniformly(number_of_points=10000)  # 根据mesh进行采样
            vertices_visual = np.asarray(obj_pcd.points)  # 获得顶点 用于可视化

            model_info_path = '/home/shenshougang/PythonProjects/Coordinate-3D/data/LINEMOD/{}/model_info_{}.json'.format(cls, cls)
            with open(model_info_path, 'r') as model_info_file:
                model_info_s = json.load(model_info_file)
            model_info = model_info_s[cls]
            x_max = model_info['max_x']  # 模型不是严格与世界原点对其的
            x_min = model_info['min_x']
            y_max = model_info['max_y']
            y_min = model_info['min_y']
            z_max = model_info['max_z']
            z_min = model_info['min_z']

            corner = np.array([[model_info['min_x'], model_info['min_y'], model_info['min_z']],
                               [model_info['min_x'], model_info['min_y'], model_info['max_z']],
                               [model_info['min_x'], model_info['max_y'], model_info['min_z']],
                               [model_info['min_x'], model_info['max_y'], model_info['max_z']],
                               [model_info['max_x'], model_info['min_y'], model_info['min_z']],
                               [model_info['max_x'], model_info['min_y'], model_info['max_z']],
                               [model_info['max_x'], model_info['max_y'], model_info['min_z']],
                               [model_info['max_x'], model_info['max_y'], model_info['max_z']]])

            cls_idx = linemod_cls_names.index(cls)
            K = cfg.linemod_K.copy()
            K[0, 2] += begins[cls_idx, 1]
            K[1, 2] += begins[cls_idx, 0]

            select_coords = coords[mask == cls_num, :]  # [n, 3]

            select_coords[:, 0] = 2.0 * select_coords[:, 0] / 255 - 1
            select_coords[:, 1] = 2.0 * select_coords[:, 1] / 255 - 1
            select_coords[:, 2] = 2.0 * select_coords[:, 2] / 255 - 1

            select_coords[:, 0] = select_coords[:, 0] * ((x_max - x_min) / 2) + (x_max + x_min) / 2
            select_coords[:, 1] = select_coords[:, 1] * ((y_max - y_min) / 2) + (y_max + y_min) / 2
            select_coords[:, 2] = select_coords[:, 2] * ((z_max - z_min) / 2) + (z_max + z_min) / 2

            img_coords = xymap[mask == cls_num, :]  # [n, 2]

            pose_from_coords = pnp(select_coords, img_coords, K)

            corner_2d_gt = project_K(corner, poses[cls_idx], K)
            corner_2d_pr = project_K(corner, pose_from_coords, K)
            plot_box_img = plot_box(img_path, corner_2d_gt, corner_2d_pr, True)

            vertices_global = o3d.geometry.PointCloud()
            vertices_global.points = o3d.utility.Vector3dVector(vertices_visual)
            vertices_global.paint_uniform_color([0, 1, 0])  # 对象所有点绘制为绿色

            vertices_partial_pr = o3d.geometry.PointCloud()
            vertices_partial_pr.points = o3d.utility.Vector3dVector(select_coords)
            vertices_partial_pr.paint_uniform_color([0, 0, 1])  # 预测的对象所有点绘制为蓝色

            o3d.visualization.draw_geometries(
                geometry_list=[vertices_global, vertices_partial_pr]
            )






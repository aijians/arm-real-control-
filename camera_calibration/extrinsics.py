"""Known-board world pose -> camera pose. All detection pixels are sensor_raw.

PnP is solved in the right-handed board frame. A separate orthogonal basis
maps board coordinates into the user's right- or left-handed world frame.
"""
import hashlib
import json

import cv2
import numpy as np


def vector(value, shape, name):
    try:
        x = np.asarray(value, dtype=np.float64).reshape(shape)
    except (ValueError, TypeError):
        raise ValueError(name + '格式错误')
    if not np.isfinite(x).all():
        raise ValueError(name + '必须为有限数值')
    return x


def load_intrinsics(data, device, sensor):
    if not isinstance(data, dict):
        raise ValueError('请上传原始内参 JSON 报告')
    # Exact physical device binding is required; transferring another camera's
    # intrinsics must not silently produce a seemingly calibrated world pose.
    source_device = data.get('device', data.get('camera_device'))
    if source_device != device or data.get('sensor') != sensor:
        raise ValueError('内参所属设备/传感器与当前相机不同，请对当前相机重新标定')
    if [data.get('image_width'), data.get('image_height')] != [1920, 1080]:
        raise ValueError('仅接受本程序1920×1080原始模式内参，不自动缩放')
    if data.get('pixel_coordinate_frame', 'sensor_raw') != 'sensor_raw':
        raise ValueError('外参采集使用原始图像；请导入sensor_raw内参')
    if data.get('pixels_are_undistorted', False):
        raise ValueError('请导入原始图像内参，不接受已去畸变像素约定')
    if data.get('distortion_model', 'plumb_bob') != 'plumb_bob':
        raise ValueError('仅支持OpenCV普通针孔畸变模型')
    K = vector(data.get('camera_matrix'), (3, 3), 'K')
    D = vector(data.get('distortion_coefficients'), (-1,), 'D')
    if D.size != 5:
        raise ValueError('本工具导出与机械臂一致的五参数畸变模型，D必须为5项')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9:
        raise ValueError('K不是支持的相机矩阵')
    result = dict(camera_matrix=K.tolist(), distortion_coefficients=D.tolist(),
                  image_width=1920, image_height=1080, device=device, sensor=sensor,
                  distortion_model='plumb_bob', pixel_coordinate_frame='sensor_raw',
                  pixels_are_undistorted=False, rms_px=data.get('rms_px'),
                  source_created_utc=data.get('created_utc'),
                  source_sample_count=data.get('sample_count'))
    result['source_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def ordered_corners(corners, config, ids):
    rows, cols = config['rows'], config['cols']
    if len(ids) != 3 or any(isinstance(i, bool) or not isinstance(i, int) for i in ids) or len(set(ids)) != 3:
        raise ValueError('依次点选三个不同的O、U、V内角点')
    if any(i < 0 or i >= rows*cols for i in ids):
        raise ValueError('角点编号越界')
    (r, c), (ru, cu), (rv, cv) = [divmod(i, cols) for i in ids]
    if r not in (0, rows-1) or c not in (0, cols-1) or ru != r or cv != c or abs(cu-c) != cols-1 or abs(rv-r) != rows-1:
        raise ValueError('O/U/V必须分别为打印图上的原点、同一行末端、同一列末端内角点；请重新点选')
    grid = vector(corners, (rows, cols, 2), '角点')
    rr = range(rows) if r == 0 else range(rows-1, -1, -1)
    cc = range(cols) if c == 0 else range(cols-1, -1, -1)
    return grid[list(rr)][:, list(cc)].reshape(-1, 1, 2)


def world_board_pose(config, data):
    O, U, V = [vector(data.get(k), (3,), k+'世界坐标') for k in ('origin', 'u_end', 'v_end')]
    du, dv = U-O, V-O
    nu, nv = np.linalg.norm(du), np.linalg.norm(dv)
    eu = (config['cols']-1)*config['square_mm']
    ev = (config['rows']-1)*config['square_mm']
    if nu < 1e-8 or nv < 1e-8 or abs(nu-eu) > max(.5, eu*.02) or abs(nv-ev) > max(.5, ev*.02):
        raise ValueError('O→U应为%.3fmm，O→V应为%.3fmm；填写内角点而非棋盘外边缘的坐标' % (eu, ev))
    a, b = du/nu, dv/nv
    cosine = float(a @ b)
    if abs(cosine) > .02:
        raise ValueError('O→U与O→V需互相垂直，请核对三点世界位置')
    b = b-a*cosine
    b /= np.linalg.norm(b)
    handedness = data.get('world_handedness')
    if handedness not in ('left', 'right'):
        raise ValueError('请选择世界坐标系手性')
    normal = np.cross(a, b) * (-1 if handedness == 'left' else 1)
    basis = np.column_stack((a, b, normal))
    return O, basis, dict(origin=O.tolist(), u_end=U.tolist(), v_end=V.tolist(),
        world_handedness=handedness, measured_u_length_mm=float(nu), measured_v_length_mm=float(nv),
        expected_u_length_mm=eu, expected_v_length_mm=ev, measured_axes_cosine=cosine,
        note='O为第一个内角点，U/V为同一行/列末内角点；使用填写格距的刚性正交棋盘，测量小误差不作剪切/伸缩拟合')


def solve_world_pose(config, corners, ids, data, intrinsics):
    plane = float(data['plane_y_mm'])
    lip = float(data['rail_lip_y_mm'])
    if not np.isfinite([plane, lip]).all():
        raise ValueError('导轨轴心/挡边平面Y必须为有限数')
    pts = ordered_corners(corners, config, ids)
    O, W_B, measurement = world_board_pose(config, data)
    cols, rows, size = config['cols'], config['rows'], config['square_mm']
    obj = np.zeros((cols*rows, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)*size
    K, D = np.array(intrinsics['camera_matrix']), np.array(intrinsics['distortion_coefficients'])
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, pts, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        raise ValueError('无法求解平面棋盘位姿')
    solutions = []
    for rv, tv in zip(rvecs, tvecs):
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():
            continue
        R, _ = cv2.Rodrigues(rv)
        depths = (obj @ R.T + tv.reshape(3))[:, 2]
        if depths.min() <= 0:
            continue
        projected, _ = cv2.projectPoints(obj, rv, tv, K, D)
        errors = np.linalg.norm(projected.reshape(-1, 2)-pts.reshape(-1, 2), axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        C_B = -R.T @ tv.reshape(3)
        C_W = O+W_B @ C_B
        W_C = W_B @ R.T
        S = np.diag([-1., -1., 1.])  # raw pixels -> 180-degree upright camera axes
        Ku, Du = K.copy(), D.copy()
        Ku[0, 2], Ku[1, 2] = 1919-K[0, 2], 1079-K[1, 2]
        Du[2:4] *= -1
        # Raw and upright variants keep identical physical C; only camera axes
        # and intrinsic image coordinates change. Never rotate the image twice.
        common = dict(name='known_world_checkerboard_candidate', image_size_px=[1920, 1080], C_world_mm=C_W.tolist(),
            distortion_model='plumb_bob', pixels_are_undistorted=False,
            world_handedness=measurement['world_handedness'],
            world_coordinate_frame=data.get('world_frame', 'measured_world_x_reversed_20261117'),
            plane_y_mm=plane, rail_lip_y_mm=lip,
            intrinsics_device=intrinsics['device'], intrinsics_sensor=intrinsics['sensor'],
            intrinsics_source_sha256=intrinsics['source_sha256'],
            is_approximate=True, source='known_world_checkerboard_fixed_intrinsics',
            validation_status='reprojection_only_independent_world_validation_required')
        raw = dict(common, K=K.tolist(), D=D.tolist(), R_world_from_camera=W_C.tolist(), pixel_coordinate_frame='sensor_raw')
        upright = dict(common, K=Ku.tolist(), D=Du.tolist(), R_world_from_camera=(W_C @ S).tolist(), pixel_coordinate_frame='scene_upright')
        solutions.append(dict(C_world_mm=C_W.tolist(), R_world_from_camera=W_C.tolist(),
            camera_optical_axis_world=W_C[:, 2].tolist(), rvec_board_to_camera=rv.ravel().tolist(),
            tvec_board_to_camera_mm=tv.ravel().tolist(), rms_px=rms, max_error_px=float(errors.max()),
            projected_pixels=projected.reshape(-1, 2).tolist(), ordered_pixels=pts.reshape(-1, 2).tolist(),
            calibration_sensor_raw=raw, calibration_scene_upright=upright))
    solutions.sort(key=lambda x: x['rms_px'])
    if not solutions:
        raise ValueError('棋盘位于相机后方或位姿无效，请检查角点和内参')
    ambiguous = len(solutions)>1 and solutions[1]['rms_px']-solutions[0]['rms_px'] < .15
    warnings = ['像素重投影通过不代表世界毫米精度通过；请使用另一已知位置的棋盘/独立点验证。']
    if ambiguous:
        warnings.append('平面棋盘存在近似双解，请对比两个光心/光轴，并用另一已知摆放位置验证；不自动应用。')
    if solutions[0]['rms_px']>1:
        warnings.append('最佳解重投影RMS超过1px，请检查棋盘尺寸、角点、清晰度及本机内参。')
    return dict(board=config, board_world_measurement=measurement, selected_corner_ids=ids,
        intrinsics=intrinsics, candidates=solutions, planar_ambiguity=ambiguous,
        warnings=warnings, coordinate_units='mm', pixel_coordinate_frame='sensor_raw',
        applied_to_robot=False)

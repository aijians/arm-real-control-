import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

import app
from extrinsics import load_intrinsics, ordered_corners, solve_world_pose


class WorldPoseTests(unittest.TestCase):
    def fixture(self, handedness='left'):
        cfg = dict(cols=8, rows=6, square_mm=6.)
        K = np.array([[700., 0, 960.], [0, 705., 540.], [0, 0, 1.]])
        D = np.array([.05, -.06, .001, -.002, .01])
        source = dict(camera_matrix=K.tolist(), distortion_coefficients=D.tolist(),
            device='/dev/video31', sensor=app.CAMERAS['/dev/video31'],
            image_width=1920, image_height=1080, sample_count=30)
        intr = load_intrinsics(source, source['device'], source['sensor'])
        W_B = cv2.Rodrigues(np.array([.1, .2, -.3]))[0]
        if handedness == 'left':
            W_B = np.diag([-1., 1., 1.]) @ W_B
        O = np.array([100., 180., -100.])
        data = dict(origin=O.tolist(), u_end=(O+W_B[:, 0]*42).tolist(),
            v_end=(O+W_B[:, 1]*30).tolist(), world_handedness=handedness,
            plane_y_mm=181.5, rail_lip_y_mm=172.5)
        rv, tv = np.array([.2, -.3, 2.9]), np.array([-20., -15., 180.])
        obj = app.object_points(cfg).astype(float)
        corners = cv2.projectPoints(obj, rv, tv, K, D)[0]
        R = cv2.Rodrigues(rv)[0]
        C = O-W_B @ R.T @ tv
        return cfg, intr, data, corners, obj, W_B, R, C

    def test_known_pose_left_and_right_handed_world(self):
        for hand in ('left', 'right'):
            cfg, intr, data, pts, obj, W_B, R, C = self.fixture(hand)
            result = solve_world_pose(cfg, pts, [0, 7, 40], data, intr)
            best = result['candidates'][0]
            np.testing.assert_allclose(best['C_world_mm'], C, atol=1e-5)
            np.testing.assert_allclose(best['R_world_from_camera'], W_B @ R.T, atol=1e-7)
            self.assertLess(best['rms_px'], 1e-7)
            self.assertFalse(result['applied_to_robot'])
            self.assertAlmostEqual(np.linalg.det(best['R_world_from_camera']), -1 if hand=='left' else 1)

    def test_all_four_corner_orderings_same_physical_pose(self):
        cfg, intr, data, pts, _, _, _, C = self.fixture()
        grid=pts.reshape(6, 8, 2)
        for rr, cc, ids in [(False, False, [0,7,40]), (True, False, [40,47,0]),
                            (False, True, [7,0,47]), (True, True, [47,40,7])]:
            reversed_grid=grid[::-1 if rr else 1, ::-1 if cc else 1]
            result=solve_world_pose(cfg, reversed_grid.reshape(-1,1,2), ids, data, intr)
            np.testing.assert_allclose(result['candidates'][0]['C_world_mm'], C, atol=1e-5)

    def test_raw_and_rotated_exports_project_same_world_points(self):
        cfg, intr, data, pts, obj, W_B, _, _ = self.fixture()
        best=solve_world_pose(cfg,pts,[0,7,40],data,intr)['candidates'][0]
        world=np.array(data['origin'])+obj @ W_B.T
        for frame in ('sensor_raw','scene_upright'):
            cal=best['calibration_'+frame]
            camera=(world-np.array(cal['C_world_mm'])) @ np.array(cal['R_world_from_camera'])
            projected=cv2.projectPoints(camera, np.zeros(3), np.zeros(3), np.array(cal['K']), np.array(cal['D']))[0]
            expected=pts if frame=='sensor_raw' else np.array([1919,1079])-pts
            np.testing.assert_allclose(projected,expected,atol=1e-6)
        # Verify exact exported schema loads in the current robot geometry code.
        import importlib.util
        path=Path(__file__).resolve().parents[2]/'vision_tools'/'pixel_to_world.py'
        if path.exists():
            spec=importlib.util.spec_from_file_location('world_pixel', path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module._calibration(best['calibration_scene_upright'])

    def test_wrong_device_mode_rotation_and_invalid_intrinsics_rejected(self):
        _, intr, _, _, _, _, _, _ = self.fixture()
        for changes in [dict(device='/dev/video22'),dict(sensor='other'),dict(image_width=960),
                        dict(pixel_coordinate_frame='scene_upright'),dict(pixels_are_undistorted=True),
                        dict(distortion_coefficients=[0]*12),dict(camera_matrix=[[float('nan')]*3]*3)]:
            with self.assertRaises(ValueError):
                load_intrinsics(dict(intr, **changes),'/dev/video31',app.CAMERAS['/dev/video31'])

    def test_invalid_world_geometry_and_correspondences_rejected(self):
        cfg, intr, data, pts, *_ = self.fixture()
        for changes in [dict(u_end=data['origin']),dict(v_end=data['u_end']),
                        dict(origin=[float('nan'),0,0]),dict(world_handedness='unknown')]:
            with self.assertRaises(ValueError):
                solve_world_pose(cfg,pts,[0,7,40],dict(data,**changes),intr)
        for ids in ([0,1,8],[0,7,7],[0,40,7],[-1,7,40],[0,True,40]):
            with self.assertRaises(ValueError):
                ordered_corners(pts,cfg,ids)

    def test_capture_export_immutable_evidence_and_new_capture_invalidates(self):
        cfg,intr,data,pts,*_=self.fixture()
        with tempfile.TemporaryDirectory() as root:
            state=app.Calibration(root,'/dev/video31',app.CAMERAS['/dev/video31'])
            state.session=Path(root)/'session';state.session.mkdir()
            state.config=cfg;state.intrinsics=intr;state.state='capturing'
            state.latest=dict(image=np.full((1080,1920,3),230,np.uint8),corners=pts,
                time=time.monotonic(),sharpness=100.,stable=2,board=cfg)
            state.freeze_extrinsics();token=state.frozen['token']
            state.compute_extrinsics(dict(data,token=token,corner_ids=[0,7,40]))
            state.export_extrinsics(dict(token=token,candidate=0))
            old=state.extrinsic_directory
            saved=json.loads((old/'camera_world_scene_upright.json').read_text())
            self.assertEqual(saved['pixel_coordinate_frame'],'scene_upright')
            self.assertTrue((old/'sensor_raw.png').exists())
            with self.assertRaises(ValueError):
                state.compute_extrinsics(dict(data,token=token,corner_ids=[0,1,8]))
            self.assertIsNone(state.exported_candidate)
            self.assertIsNone(state.extrinsic_result)
            state.freeze_extrinsics()
            self.assertNotEqual(state.frozen['token'],token)
            self.assertTrue((old/'extrinsics.json').exists())
            with self.assertRaises(ValueError):
                state.compute_extrinsics(dict(data,token=token,corner_ids=[0,7,40]))
            state.new_session()
            self.assertIsNone(state.frozen)
            self.assertTrue(old.exists())

    def test_stale_blurred_or_moving_frame_rejected(self):
        cfg,intr,data,pts,*_=self.fixture()
        with tempfile.TemporaryDirectory() as root:
            state=app.Calibration(root,'/dev/video31',app.CAMERAS['/dev/video31'])
            state.state='capturing';state.intrinsics=intr
            for changes in (dict(time=time.monotonic()-3),dict(sharpness=0),dict(stable=0),dict(corners=None)):
                state.latest=dict(image=np.zeros((1080,1920,3),np.uint8),corners=pts,
                    time=time.monotonic(),sharpness=100,stable=2,board=cfg,**{})
                state.latest.update(changes)
                with self.assertRaises(ValueError):state.freeze_extrinsics()

    def test_switch_camera_clears_intrinsics_and_retains_files(self):
        with tempfile.TemporaryDirectory() as root:
            state=app.Calibration(root,'/dev/video22',app.CAMERAS['/dev/video22'])
            old=Path(root)/'old';old.mkdir();(old/'test.txt').write_text('retained')
            state.intrinsics={'old':True}
            state.set_device({'device':'/dev/video31'})
            self.assertEqual(state.device,'/dev/video31')
            self.assertIsNone(state.intrinsics)
            self.assertIsNone(state.worker)
            self.assertTrue((old/'test.txt').exists())


if __name__=='__main__':
    unittest.main()

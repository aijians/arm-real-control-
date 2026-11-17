import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('calibration_app', Path(__file__).resolve().parents[1] / 'app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class CalibrationTests(unittest.TestCase):
    def test_validate_physical_board_and_units(self):
        for bad in (0, float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                app.board_config(dict(cols=9, rows=6, square_mm=bad))
        obj = app.object_points(dict(cols=9, rows=6, square_mm=20))
        np.testing.assert_array_equal(obj[-1], [160, 100, 0])

    def test_recover_camera_with_distortion_and_pose_variation(self):
        config = dict(cols=9, rows=6, square_mm=20)
        K = np.array([[900., 0, 960], [0, 890., 540], [0, 0, 1.]])
        D = np.array([-.16, .035, .001, -.002, .004])
        points = app.object_points(config)
        corners, metadata = [], []
        for i in range(30):
            r = np.array([.35 * np.sin(i), .4 * np.cos(i * .7), .15 * np.sin(i * .4)])
            t = np.array([-100. + (i % 5) * 20, -70. + (i % 4) * 15, 300. + (i % 6) * 30])
            c, _ = cv2.projectPoints(points, r, t, K, D)
            corners.append(c)
            metadata.append(dict(cell=i % 9))
        result = app.solve(config, corners, metadata)
        np.testing.assert_allclose(result['camera_matrix'], K, atol=.03)
        np.testing.assert_allclose(result['distortion_coefficients'], D, atol=.001)
        self.assertLess(result['rms_px'], .001)
        with self.assertRaises(ValueError):
            app.solve(config, corners[:5], metadata[:5])

    def test_detect_printable_board(self):
        gray = np.full((1080, 1920), 180, np.uint8)
        for y in range(7):
            for x in range(10):
                gray[250+y*60:250+(y+1)*60, 500+x*60:500+(x+1)*60] = 255 if (x+y) % 2 else 0
        found, corners = cv2.findChessboardCorners(cv2.resize(gray, (960, 540)), (9, 6))
        self.assertTrue(found)
        refined = cv2.cornerSubPix(gray, np.float32(corners*2), (5, 5), (-1, -1), (3, 40, .001))
        self.assertEqual(refined.shape, (54, 1, 2))

    def test_idle_no_camera_and_new_group_retains_old_files(self):
        with tempfile.TemporaryDirectory() as root:
            state = app.Calibration(root, '/dev/video22', 'test_camera')
            self.assertIsNone(state.worker)
            self.assertEqual(state.status()['state'], 'idle')
            state.session = Path(root) / 'old'
            state.session.mkdir()
            state.config = dict(cols=9, rows=6, square_mm=20)
            state.samples = [dict(file='sample_001.png', cell=4)]
            state.manifest()
            state.new_session()
            self.assertTrue((Path(root) / 'old' / 'samples.json').exists())
            self.assertEqual(state.samples, [])
            for action in ('auto', 'take', 'calibrate', 'corrected'):
                with self.assertRaises(ValueError):
                    state.action(action, {})

    def test_result_yaml_and_json_are_consistent(self):
        config = dict(cols=9, rows=6, square_mm=20)
        K = np.array([[900., 0, 960], [0, 890., 540], [0, 0, 1.]])
        obj = app.object_points(config)
        corners = [cv2.projectPoints(obj, np.array([.1*np.sin(i), .3*np.cos(i), 0.]),
                    np.array([-80., -50., 300.+10*i]), K, None)[0] for i in range(25)]
        with tempfile.TemporaryDirectory() as root:
            state = app.Calibration(root, '/dev/video22', 'test_camera')
            state._solve(corners, [dict(cell=i % 9, file=f'{i}.png') for i in range(25)], config, Path(root))
            self.assertEqual(state.state, 'complete', state.message)
            result = json.loads((Path(root) / 'calibration.json').read_text())
            fs = cv2.FileStorage(str(Path(root) / 'calibration.yaml'), cv2.FILE_STORAGE_READ)
            np.testing.assert_allclose(fs.getNode('camera_matrix').mat(), result['camera_matrix'])
            fs.release()
            self.assertEqual(result['sensor'], 'test_camera')


if __name__ == '__main__':
    unittest.main()

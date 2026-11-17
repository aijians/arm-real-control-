from pathlib import Path
import json,numpy as np,hashlib
out=Path(__file__).parent
cal=json.loads(Path('/home/aijians/下载/calibration (2).json').read_text());ext=json.loads(Path('/home/aijians/下载/extrinsics.json').read_text())
for src,dst in [('/home/aijians/下载/calibration (2).json','input_intrinsics.json'),('/home/aijians/下载/extrinsics.json','input_extrinsics.json')]:
 (out/dst).write_bytes(Path(src).read_bytes())
h=hashlib.sha256(json.dumps(cal,sort_keys=True,allow_nan=False).encode()).hexdigest();a=np.array(cal['per_image_rms_px']);s=np.sum(a*a)
summary=dict(intrinsics=dict(device=cal['device'],sensor=cal['sensor'],rms_px=cal['rms_px'],median_image_rms_px=float(np.median(a)),worst_images=sorted(zip(cal['files'],a.tolist()),key=lambda x:x[1],reverse=True)[:5],worst_squared_error_fraction=float(a.max()**2/s),coverage=cal['coverage_cells'],canonical_sha256=h,external_report_same_source=h==ext['intrinsics']['source_sha256'],equal_K=np.array_equal(cal['camera_matrix'],ext['intrinsics']['camera_matrix']),equal_D=np.array_equal(cal['distortion_coefficients'],ext['intrinsics']['distortion_coefficients'])),world=ext['board_world_measurement'],corner_ids=ext['selected_corner_ids'],candidates=[])
nominal=np.array([228.99,97,-104.22])
for i,c in enumerate(ext['candidates']):
 R=np.array(c['R_world_from_camera']);C=np.array(c['C_world_mm']);axis=R[:,2];err=np.linalg.norm(np.array(c['ordered_pixels'])-c['projected_pixels'],axis=1)
 summary['candidates'].append(dict(index=i,C_world_mm=C.tolist(),camera_optical_axis_world=axis.tolist(),rms_px=float(np.sqrt(np.mean(err**2))),max_error_px=float(err.max()),off_nominal_axis_deg=float(np.degrees(np.arccos(np.clip(-axis[0],-1,1)))),delta_vs_measured_lens_point_mm=(C-nominal).tolist(),distance_vs_measured_lens_point_mm=float(np.linalg.norm(C-nominal)),det=float(np.linalg.det(R)),orthogonality_error=float(np.max(np.abs(R.T@R-np.eye(3))))))
summary['candidate_rms_ratio']=summary['candidates'][1]['rms_px']/summary['candidates'][0]['rms_px']
summary['candidate_center_distance_mm']=float(np.linalg.norm(np.subtract(ext['candidates'][0]['C_world_mm'],ext['candidates'][1]['C_world_mm'])))
(out/'summary_initial.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

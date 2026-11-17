"""Independent geometry/PnP check from saved raw pixels. No hardware access."""
from pathlib import Path
import hashlib,json,math
import cv2,numpy as np
cv2.setNumThreads(1)
out=Path(__file__).resolve().parent
ext=json.loads((out/'provided_extrinsics.json').read_text())
raw=json.loads((out/'provided_camera_world_sensor_raw.json').read_text())
intr=json.loads((out/'provided_intrinsics.json').read_text())
cap=json.loads((out/'board_evidence/capture.json').read_text())
assert cap['image_sha256']==hashlib.sha256((out/'board_evidence/sensor_raw.png').read_bytes()).hexdigest()
assert ext['intrinsics']==cap['intrinsics']
assert raw==ext['candidates'][0]['calibration_sensor_raw']
assert raw['K']==intr['camera_matrix'] and raw['D']==intr['distortion_coefficients']
assert raw['image_size_px']==[1920,1080] and raw['pixel_coordinate_frame']=='sensor_raw'
board=ext['board'];rows,cols=board['rows'],board['cols'];size=board['square_mm']
ids=ext['selected_corner_ids'];assert ids==[47,40,7]
observed=np.array(cap['corners'],float).reshape(rows,cols,2)[::-1,::-1].reshape(-1,2)
obj=np.zeros((rows*cols,3),float);obj[:,:2]=np.mgrid[0:cols,0:rows].T.reshape(-1,2)*size
m=ext['board_world_measurement'];O,U,V=[np.array(m[k],float) for k in ['origin','u_end','v_end']]
a=(U-O)/np.linalg.norm(U-O);b=(V-O)/np.linalg.norm(V-O)
B=np.column_stack((a,b,-np.cross(a,b)))
assert np.allclose(B.T@B,np.eye(3),atol=1e-12) and abs(np.linalg.det(B)+1)<1e-12
world=O+obj@B.T
K=np.array(raw['K']);D=np.array(raw['D']);S=np.diag([-1.,-1.,1.])
ok,rv,tv,_=cv2.solvePnPGeneric(obj,observed,K,D,flags=cv2.SOLVEPNP_IPPE)
assert ok
independent=[]
for r,t in zip(rv,tv):
 rc=cv2.Rodrigues(r)[0];pc=obj@rc.T+t.ravel()
 if pc[:,2].min()<=0:continue
 pr=cv2.projectPoints(obj,r,t,K,D)[0].reshape(-1,2)
 err=np.linalg.norm(pr-observed,axis=1)
 independent.append(dict(C=O+B@(-rc.T@t.ravel()),R=B@rc.T,rms=float(np.sqrt(np.mean(err**2)))))
independent.sort(key=lambda c:c['rms'])
assert len(independent)==len(ext['candidates'])==2
results=[]
for index,c in enumerate(ext['candidates']):
 C=np.array(c['C_world_mm']);R=np.array(c['R_world_from_camera']);pc=(world-C)@R
 pr=cv2.projectPoints(pc,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
 e=np.linalg.norm(pr-observed,axis=1);rms=float(np.sqrt(np.mean(e**2)))
 assert np.max(np.abs(np.array(c['ordered_pixels'])-observed))<1e-12
 assert np.max(np.abs(np.array(c['projected_pixels'])-pr))<1e-8
 assert abs(rms-c['rms_px'])<1e-9
 assert np.max(np.abs(C-independent[index]['C']))<1e-7
 assert np.max(np.abs(R-independent[index]['R']))<1e-9
 assert abs(np.linalg.det(R)+1)<1e-12
 u=c['calibration_scene_upright'];ku=K.copy();ku[0,2]=1919-K[0,2];ku[1,2]=1079-K[1,2]
 du=D.copy();du[2:4]*=-1
 assert u['C_world_mm']==c['C_world_mm'] and np.array_equal(u['K'],ku) and np.array_equal(u['D'],du)
 assert np.array_equal(u['R_world_from_camera'],R@S)
 tilt=math.degrees(math.acos(float(np.clip(-R[0,2],-1,1))))
 results.append(dict(candidate=index,C_world_mm=C.tolist(),axis_world=R[:,2].tolist(),
  optical_axis_tilt_from_world_minus_X_deg=tilt,rms_px=rms,max_error_px=float(e.max()),
  min_depth_mm=float(pc[:,2].min()),max_depth_mm=float(pc[:,2].max()),
  orthogonality_max_error=float(np.max(np.abs(R.T@R-np.eye(3)))),determinant=float(np.linalg.det(R)),
  independently_reproduced_from_raw_corners=True,upright_conversion_consistent=True))
old=json.loads((out.parent/'new_calibration_photo_test_20261117/provided_camera_world_sensor_raw.json').read_text())
result=dict(candidates=results,report_matches_downloaded_board_evidence=True,provided_export_is_candidate=0,
 intrinsic_values_match_reviewed_v3=True,O=O.tolist(),U=U.tolist(),V=V.tolist(),
 OU_mm=float(np.linalg.norm(U-O)),OV_mm=float(np.linalg.norm(V-O)),axes_dot=float(a@b),
 clockwise_corner_ids=ids,board_world_pose_independently_measured=False,
 rms_ratio_candidate1_to_0=results[1]['rms_px']/results[0]['rms_px'],
 candidate_center_separation_mm=float(np.linalg.norm(np.array(results[1]['C_world_mm'])-results[0]['C_world_mm'])),
 candidate0_vs_previous_center_delta_mm=(np.array(raw['C_world_mm'])-old['C_world_mm']).tolist(),
 rail_axis_plane_y_mm=raw['plane_y_mm'],rail_lip_plane_y_mm=raw['rail_lip_y_mm'],
 previous_rail_lip_y_mm=old['rail_lip_y_mm'],not_applied_to_robot=True)
(out/'extrinsics_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))

from pathlib import Path
import json,sys,hashlib
import cv2,numpy as np
out=Path(__file__).parent; ev=out/'board_evidence'
cal=json.loads((out/'input_intrinsics.json').read_text());ext=json.loads((out/'input_extrinsics.json').read_text());samples=json.loads((ev/'samples.json').read_text());cap=json.loads((ev/'world_pose_f471d02363658eb2/capture.json').read_text())
assert (ev/'calibration.json').read_bytes()==(out/'input_intrinsics.json').read_bytes()
assert len(samples['samples'])==30
assert cap['image_sha256']==hashlib.sha256((ev/'world_pose_f471d02363658eb2/sensor_raw.png').read_bytes()).hexdigest()
assert cal['files']==[s['file'] for s in samples['samples']]
obj=np.zeros((48,3),np.float32);obj[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)*6
pts=[np.float32(s['corners']).reshape(-1,1,2) for s in samples['samples']]
def fit(ids):
 rms,K,D,rs,ts=cv2.calibrateCamera([obj for i in ids],[pts[i] for i in ids],(1920,1080),None,None)
 return dict(rms_px=float(rms),K=K.tolist(),D=D.ravel().tolist()),rs,ts
base,rs,ts=fit(list(range(30)))
assert np.allclose(base['K'],cal['camera_matrix'],atol=1e-7)
assert np.allclose(base['D'],cal['distortion_coefficients'],atol=1e-7)
fullK=np.array(base['K']);fullD=np.array(base['D'])
items=[]
for i,(p,r,t) in enumerate(zip(pts,rs,ts)):
 projected=cv2.projectPoints(obj,r,t,fullK,fullD)[0]
 errors=np.linalg.norm(projected.reshape(-1,2)-p.reshape(-1,2),axis=1)
 items.append(dict(file=samples['samples'][i]['file'],rms_px=float(np.sqrt(np.mean(errors**2))),max_error_px=float(errors.max()),worst_corner=int(np.argmax(errors)),center_px=p.mean(axis=0).ravel().tolist(),bounds_px=[p.min(axis=0).ravel().tolist(),p.max(axis=0).ravel().tolist()],tvec=t.ravel().tolist(),rvec=r.ravel().tolist()))
without18,_,_=fit([i for i in range(30) if i!=17])
# Diagnostic sensitivity only. Never write this refit as an active intrinsic calibration.
sys.path.insert(0,str(Path('arm_real_control/camera_calibration').resolve()))
from extrinsics import solve_world_pose,load_intrinsics,ordered_corners
world=ext['board_world_measurement'];data={**world,'plane_y_mm':181.5,'rail_lip_y_mm':172.5}
rep=solve_world_pose(ext['board'],np.array(cap['corners']),ext['selected_corner_ids'],data,ext['intrinsics'])
assert np.allclose(rep['candidates'][0]['C_world_mm'],ext['candidates'][0]['C_world_mm'],atol=1e-7)
sensitivity=load_intrinsics(dict(cal,camera_matrix=without18['K'],distortion_coefficients=without18['D']),cal['device'],cal['sensor'])
check=solve_world_pose(ext['board'],np.array(cap['corners']),ext['selected_corner_ids'],data,sensitivity)
bestC=np.array(ext['candidates'][0]['C_world_mm']);diagC=np.array(check['candidates'][0]['C_world_mm'])
# Independently verify reprojection from world C/R, including handedness.
O=np.array(world['origin']);U=np.array(world['u_end']);V=np.array(world['v_end']);a=(U-O)/42;b=(V-O)/30;B=np.column_stack((a,b,-np.cross(a,b)));PW=O+obj.astype(float)@B.T
outcomes=[]
for index,c in enumerate(ext['candidates']):
 C=np.array(c['C_world_mm']);R=np.array(c['R_world_from_camera']);PC=(PW-C)@R
 projected=cv2.projectPoints(PC,np.zeros(3),np.zeros(3),fullK,fullD)[0]
 err=np.linalg.norm(projected.reshape(-1,2)-np.array(c['ordered_pixels']),axis=1)
 outcomes.append(dict(index=index,rms_px=float(np.sqrt(np.mean(err**2))),max_error_px=float(err.max()),min_depth_mm=float(PC[:,2].min()),max_depth_mm=float(PC[:,2].max())))
flat=np.concatenate(pts).reshape(-1,2);hull=cv2.convexHull(flat.astype(np.float32));coverage=float(cv2.contourArea(hull)/(1920*1080))
result=dict(full_refit=base,per_image=items,reproduced_uploaded_intrinsics=True,reproduced_uploaded_extrinsics=True,
 pixels_bounds=[flat.min(axis=0).tolist(),flat.max(axis=0).tolist()],all_corner_convex_hull_fraction=coverage,
 independent_world_C_R_reprojection=outcomes,
 leave_out_18_diagnostic_only=dict(refit=without18,delta_K=(np.array(without18['K'])-fullK).tolist(),
 external_best_C_world_mm=diagC.tolist(),delta_external_C_mm=(diagC-bestC).tolist(),
 delta_external_C_norm_mm=float(np.linalg.norm(diagC-bestC)),extrinsic_rms_px=check['candidates'][0]['rms_px'],
 note='只作敏感性检查，不是已独立验证的新内参；未修改用户报告/程序/板端配置'),
 file_origin_verified=True,actual_OUV_world_position_independently_measured=False)
(out/'recalculation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_image'},ensure_ascii=False,indent=2));print('sample18',json.dumps(items[17],ensure_ascii=False))

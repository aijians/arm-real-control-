from pathlib import Path
import json,hashlib
import cv2,numpy as np
cv2.setNumThreads(2)
out=Path(__file__).parent
new=json.loads((out/'provided_calibration_v3.json').read_text());old=json.loads(Path('diagnostics/calibration_review_20261117/input_intrinsics.json').read_text());meta=json.loads((out/'board_evidence/samples.json').read_text())
assert hashlib.sha256((out/'provided_calibration_v3.json').read_bytes()).hexdigest()=='c79b71c47d790c6f1fc68ab29819366aefd5b82a5b138bd35e6bcbb426a7fa36'
assert json.loads((out/'board_evidence/calibration.json').read_text())==new
assert new['files']==[m['file'] for m in meta['samples']] and meta['board']==new['board']
points=[np.array(m['corners'],np.float32).reshape(-1,1,2) for m in meta['samples']];n=len(points)
obj=np.zeros((48,3),np.float32);obj[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)*6

def fit(ids):
 rms,K,D,rs,ts=cv2.calibrateCamera([obj for i in ids],[points[i] for i in ids],(1920,1080),None,None)
 return float(rms),K,D.ravel(),rs,ts
rms,K,D,rs,ts=fit(list(range(n)))
assert abs(rms-new['rms_px'])<1e-9
assert np.allclose(K,new['camera_matrix'],atol=1e-6,rtol=0)
assert np.allclose(D,new['distortion_coefficients'],atol=1e-6,rtol=0)
per=[]
for i,(r,t) in enumerate(zip(rs,ts)):
 pr=cv2.projectPoints(obj,r,t,K,D)[0];e=np.linalg.norm(pr-points[i],axis=2).ravel();R=cv2.Rodrigues(r)[0];tilt=np.degrees(np.arccos(np.clip(abs(R[2,2]),0,1)))
 per.append(dict(file=new['files'][i],cell=meta['samples'][i]['cell'],sharpness=meta['samples'][i]['sharpness'],rms_px=float(np.sqrt(np.mean(e*e))),max_error_px=float(e.max()),tilt_from_frontoparallel_deg=float(tilt),distance_z_mm=float(t[2,0]),center_px=points[i].mean(axis=0).ravel().tolist()))
folds=[]
for fold in range(5):
 train=[i for i in range(n) if i%5!=fold];test=[i for i in range(n) if i%5==fold]
 trms,k,d,_,_=fit(train);errors=[]
 for i in test:
  ok,r,t=cv2.solvePnP(obj,points[i],k,d,flags=cv2.SOLVEPNP_ITERATIVE);assert ok
  proj=cv2.projectPoints(obj,r,t,k,d)[0];errors.extend(np.linalg.norm(proj-points[i],axis=2).ravel().tolist())
 errors=np.array(errors);folds.append(dict(fold=fold,train_samples=len(train),test_files=[new['files'][i] for i in test],train_coverage=sorted(set(meta['samples'][i]['cell'] for i in train)),train_rms_px=trms,test_rms_px=float(np.sqrt(np.mean(errors**2))),test_max_error_px=float(errors.max()),K=k.tolist(),D=d.tolist()))
params=np.array([[d['K'][0][0],d['K'][1][1],d['K'][0][2],d['K'][1][2]] for d in folds])
# Stress check removing the three worst fit images, diagnostic only, not published as replacement.
worst=np.argsort(new['per_image_rms_px'])[-3:];reduced_rms,kr,dr,_,_=fit([i for i in range(n) if i not in worst])
# Reprojection convergence check for raw pixels over the whole frame.
grid=np.array([[x,y] for y in np.linspace(0,1079,13) for x in np.linspace(0,1919,21)],float).reshape(-1,1,2)
un=cv2.undistortPointsIter(grid,K,D,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12));camera=np.column_stack((un.reshape(-1,2),np.ones(len(un))))
back=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),K,D)[0];back_error=np.linalg.norm(back-grid,axis=2)
flat=np.concatenate(points).reshape(-1,2);hull=cv2.contourArea(cv2.convexHull(flat.astype(np.float32)))/(1920*1080)
counts={str(c):sum(x['cell']==c for x in meta['samples']) for c in range(9)}
get=lambda d:dict(sample_count=d['sample_count'],coverage_cells=d['coverage_cells'],rms_px=d['rms_px'],median_image_rms_px=float(np.median(d['per_image_rms_px'])),max_image_rms_px=float(max(d['per_image_rms_px'])),over_0_5_px=int(sum(x>.5 for x in d['per_image_rms_px'])),quality=d['quality'])
result=dict(input_sha256=hashlib.sha256((out/'provided_calibration_v3.json').read_bytes()).hexdigest(),device=new['device'],sensor=new['sensor'],image_size=[1920,1080],pixel_frame=new['pixel_coordinate_frame'],reproduced_from_saved_34x48_corners=True,
 old=get(old),new=get(new),delta_K=(K-np.array(old['camera_matrix'])).tolist(),focal_change_pct=((K.diagonal()[:2]/np.array(old['camera_matrix']).diagonal()[:2]-1)*100).tolist(),
 per_image=per,coverage_sample_counts=counts,pixel_corner_bounds=[flat.min(axis=0).tolist(),flat.max(axis=0).tolist()],corner_convex_hull_fraction=float(hull),
 tilt_range_deg=[min(x['tilt_from_frontoparallel_deg'] for x in per),max(x['tilt_from_frontoparallel_deg'] for x in per)],board_depth_range_mm=[min(x['distance_z_mm'] for x in per),max(x['distance_z_mm'] for x in per)],
 cross_validation_5fold=dict(folds=folds,parameter_names=['fx','fy','cx','cy'],parameter_min=params.min(axis=0).tolist(),parameter_max=params.max(axis=0).tolist(),parameter_std=params.std(axis=0).tolist(),test_rms_range_px=[min(d['test_rms_px'] for d in folds),max(d['test_rms_px'] for d in folds)],note='Images held out from intrinsic fitting; their board poses are estimated for reprojection. Not independent world-millimeter validation.'),
 leave_out_worst3_diagnostic_only=dict(excluded_files=[new['files'][i] for i in worst],rms_px=reduced_rms,delta_K=(kr-K).tolist(),D=dr.tolist(),not_applied=True),
 whole_image_pixel_roundtrip_check=dict(n=len(grid),maximum_pixel_error=float(back_error.max()),passed=bool(back_error.max()<1e-5)),
 production_modified=False,live_camera_opened=False,world_extrinsics_recomputed=False)
(out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['per_image','cross_validation_5fold']},ensure_ascii=False,indent=2));print('CV',json.dumps({k:v for k,v in result['cross_validation_5fold'].items() if k!='folds'},indent=2))

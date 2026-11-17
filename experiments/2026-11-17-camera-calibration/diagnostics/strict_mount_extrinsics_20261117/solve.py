"""Solve extrinsics with user-confirmed fixed mount and known board geometry.

Primary fit fixes K/D, Cy, Cz and optical axis exactly. Only Cx and roll vary.
Additional principal-point fits are labelled diagnostics, never deployed.
"""
from pathlib import Path
import copy,csv,hashlib,json,math,sys
import cv2,numpy as np
from scipy.optimize import least_squares
cv2.setNumThreads(1)
out=Path(__file__).resolve().parent;workspace=out.parents[1]
prev=out.parent/'new_extrinsics_v3_photo_test_20261117'
sys.path.insert(0,str(workspace/'arm_real_control'))
from vision_tools.detect_sliders import detect_frame
from vision_tools.pixel_to_world import convert_detections
raw=json.loads((prev/'provided_camera_world_sensor_raw.json').read_text())
ext=json.loads((prev/'provided_extrinsics.json').read_text())
cap=json.loads((prev/'board_evidence/capture.json').read_text())
assert hashlib.sha256((prev/'board_evidence/sensor_raw.png').read_bytes()).hexdigest()==cap['image_sha256']
assert np.array_equal(np.array(cap['corners']).reshape(6,8,2)[::-1,::-1].reshape(-1,2),ext['candidates'][0]['ordered_pixels'])
K=np.array(raw['K']);D=np.array(raw['D']);observed=np.array(ext['candidates'][0]['ordered_pixels'])
obj=np.zeros((48,3),float);obj[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)*6
O,U,V=[np.array(ext['board_world_measurement'][k],float) for k in ['origin','u_end','v_end']]
assert np.array_equal(O,[9.2,174.4,-25.8]) and np.array_equal(U,[9.2,132.4,-25.8]) and np.array_equal(V,[9.2,174.4,-55.8])
a=(U-O)/42;b=(V-O)/30;world=O+obj@np.column_stack([a,b,-np.cross(a,b)]).T
fixed_Cy,fixed_Cz=97.,-102.8
def R(phi):
 c,s=np.cos(phi),np.sin(phi)
 return np.array([[0.,0.,-1.],[c,-s,0.],[s,c,0.]])
def project(p,free_principal=False):
 C=np.array([p[0],fixed_Cy,fixed_Cz]);r=R(p[1]);pc=(world-C)@r
 k=K.copy()
 if free_principal:k[0,2]+=p[2];k[1,2]+=p[3]
 return cv2.projectPoints(pc,np.zeros(3),np.zeros(3),k,D)[0].reshape(-1,2)
def residual(p):return (project(p)-observed).ravel()
# Independent normalized-plane linear solution for a global initial estimate.
n=cv2.undistortPointsIter(observed.reshape(-1,1,2),K,D,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-13)).reshape(-1,2)
dy=world[:,1]-fixed_Cy;dz=world[:,2]-fixed_Cz
design=np.zeros((96,2));design[0::2,0]=dy;design[0::2,1]=dz;design[1::2,0]=dz;design[1::2,1]=-dy
ab=np.linalg.lstsq(design,n.ravel(),rcond=None)[0]
linear=np.array([O[0]+1/np.linalg.norm(ab),math.atan2(ab[1],ab[0])])
seeds=[linear.tolist()]+[[x,linear[1]+shift] for x in [150.,236.63,350.] for shift in [0.,np.pi/2,np.pi,-np.pi/2]]
runs=[]
for seed in seeds:
 f=least_squares(residual,seed,bounds=([O[0]+.1,-np.inf],[2000.,np.inf]),xtol=1e-13,ftol=1e-13,gtol=1e-10,max_nfev=2000)
 phi=float((f.x[1]+np.pi)%(2*np.pi)-np.pi);p=np.array([f.x[0],phi]);e=residual(p).reshape(-1,2)
 runs.append(dict(seed=seed,success=bool(f.success),Cx_mm=float(p[0]),roll_rad=phi,roll_deg=float(np.degrees(phi)),rms_px=float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),sse_px2=float(np.sum(e*e)),nfev=f.nfev))
best=min(runs,key=lambda r:r['sse_px2']);p=np.array([best['Cx_mm'],best['roll_rad']]);r=R(p[1]);C=[float(p[0]),fixed_Cy,fixed_Cz]
assert all(v['success'] for v in runs)
near=[v for v in runs if v['rms_px']<best['rms_px']+1e-5]
strict=copy.deepcopy(raw);strict.update(name='strict_fixed_mount_constrained_fit',C_world_mm=C,R_world_from_camera=r.tolist(),
 source='known_world_checkerboard_fixed_Cy_Cz_optical_axis_fit_Cx_roll',validation_status='failed_reprojection_and_independent_photo_check_not_for_activation',
 is_approximate=True,constraints={'Cy_mm':97.,'Cz_mm':-102.8,'optical_axis_world':[-1.,0.,0.],'K_D_fixed':True,'world_OUV_accepted_as_user_confirmed':True})
(out/'camera_world_sensor_raw_constrained_failed.json').write_text(json.dumps(strict,ensure_ascii=False,indent=2)+'\n')
S=np.diag([-1.,-1.,1.])
def upright(cal):
 u=copy.deepcopy(cal);k=np.array(u['K']);d=np.array(u['D']);k[0,2]=1919-k[0,2];k[1,2]=1079-k[1,2];d[2:4]*=-1
 u.update(K=k.tolist(),D=d.tolist(),R_world_from_camera=(np.array(u['R_world_from_camera'])@S).tolist(),pixel_coordinate_frame='scene_upright')
 return u
strict_u=upright(strict)
(out/'camera_world_scene_upright_constrained_failed.json').write_text(json.dumps(strict_u,ensure_ascii=False,indent=2)+'\n')
config=json.loads((prev/'used_vision_config.json').read_text());config['calibration']=strict_u;config['shoulder_global_xyz_mm']=[65.,26.5,-64.8]
(out/'offline_vision_config_failed.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
rows=[];sources=json.loads((prev/'input_manifest.json').read_text())
for source in sources:
 photo=source['photo'];assert photo in ['0001','0002','0005']
 path=Path(source['original_absolute_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
 pixels=detect_frame(cv2.imread(str(path)),profile='real',already_rotated=False)
 old=json.loads((prev/f'result_{photo}.json').read_text());assert pixels==old['pixels']
 converted=convert_detections(pixels,strict_u)
 for color in ['blue','yellow']:
  xyz=converted['sliders'][color]['world_xyz_mm'];ref=old['sliders'][color]['reference_world_x_mm']
  rows.append(dict(photo=photo,color=old['sliders'][color]['color'],reference_X_mm=ref,world_X_mm=xyz[0],error_X_mm=xyz[0]-ref,
   world_Y_mm=xyz[1],world_Z_mm=xyz[2],arm_X_mm=65-xyz[0],arm_Y_mm=xyz[1]-26.5,arm_Z_mm=xyz[2]+64.8,pixels_identical=True))
with (out/'photo_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
e=np.array([v['error_X_mm'] for v in rows]);photo_metrics=dict(n=len(e),MAE_mm=float(abs(e).mean()),RMSE_mm=float(np.sqrt(np.mean(e*e))),max_abs_error_mm=float(abs(e).max()),within_0_5_mm=int(sum(abs(e)<=.5)))

# Re-detect only saved checkerboard ROI at full resolution, independently.
image=cv2.imread(str(prev/'board_evidence/sensor_raw.png'))
lo=np.floor(observed.min(axis=0)-50).astype(int);hi=np.ceil(observed.max(axis=0)+50).astype(int)
crop=image[lo[1]:hi[1],lo[0]:hi[0]];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
ok,redetect=cv2.findChessboardCornersSB(gray,(8,6),cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
corner_check=dict(success=bool(ok),method='independent_full_resolution_SB_on_ROI')
if ok:
 grid=redetect.reshape(6,8,2).astype(float)+lo
 options=[grid,grid[::-1],grid[:,::-1],grid[::-1,::-1]]
 aligned=min([v.reshape(-1,2) for v in options],key=lambda v:np.mean((v-observed)**2))
 delta=np.linalg.norm(aligned-observed,axis=1)
 fr=least_squares(lambda q:(project(q)-aligned).ravel(),p,bounds=([O[0]+.1,-np.inf],[2000.,np.inf]),xtol=1e-13,ftol=1e-13,gtol=1e-10,max_nfev=1000)
 er=np.linalg.norm(project(fr.x)-aligned,axis=1)
 corner_check.update(RMS_difference_vs_saved_px=float(np.sqrt(np.mean(delta**2))),max_difference_px=float(delta.max()),
  strict_refit_rms_px=float(np.sqrt(np.mean(er**2))),strict_refit_Cx_mm=float(fr.x[0]),corners_sensor_raw=aligned.tolist())

# Diagnostic only: can a principal-point discrepancy explain the conflict?
# This is NOT a new intrinsic calibration and is not exported as usable K.
diag=least_squares(lambda q:(project(q,True)-observed).ravel(),[p[0],p[1],0.,0.],bounds=([O[0]+.1,-np.inf,-300.,-300.],[2000.,np.inf,300.,300.]),xtol=1e-13,ftol=1e-13,gtol=1e-10,max_nfev=2000)
de=np.linalg.norm(project(diag.x,True)-observed,axis=1)
diag_report=dict(purpose='sensitivity_only_not_a_valid_new_intrinsic_calibration',K_D_primary_fit_unchanged=True,
 success=bool(diag.success),Cx_mm=float(diag.x[0]),roll_deg=float(np.degrees(diag.x[1])),principal_delta_px=diag.x[2:].tolist(),
 hypothetical_cx_cy_px=[float(K[0,2]+diag.x[2]),float(K[1,2]+diag.x[3])],rms_px=float(np.sqrt(np.mean(de**2))),
 hypothetical_intrinsics_exported=False,single_world_pose_not_sufficient_for_new_intrinsics=True)
projected=project(p);errornorm=np.linalg.norm(projected-observed,axis=1)
corner_rows=[dict(corner=i,world_xyz_mm=world[i].tolist(),observed_px=observed[i].tolist(),strict_projected_px=projected[i].tolist(),error_px=float(errornorm[i])) for i in range(48)]
(out/'corner_residuals.json').write_text(json.dumps(corner_rows,ensure_ascii=False,indent=2)+'\n')
summary=dict(user_confirmed_constraints=strict['constraints'],world_OUV=ext['board_world_measurement'],linear_normalized_initialization=linear.tolist(),
 intrinsic_K_D_kept_identical_to_v3=True,strict_solution={'C_world_mm':C,'R_world_from_camera':r.tolist(),'optical_axis_world':[-1.,0.,0.],
 'roll_deg':float(np.degrees(p[1])),'checkerboard_RMS_px':float(np.sqrt(np.mean(errornorm**2))),'checkerboard_max_error_px':float(errornorm.max())},
 multistart={'runs':runs,'total':len(runs),'same_minimum_runs':len(near)},photo_metrics=photo_metrics,photo_results=rows,
 full_resolution_corner_recheck=corner_check,principal_point_sensitivity_only=diag_report,
 photo_reference_values_used_for_fit=False,excluded_photos=['0003','0004'],production_or_board_modified=False,live_camera_or_UART_opened=False,
 accepted_for_activation=False,source_sha256={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in [prev/'provided_camera_world_sensor_raw.json',prev/'provided_extrinsics.json',prev/'board_evidence/capture.json']})
(out/'solution_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['multistart','photo_results','source_sha256']},ensure_ascii=False,indent=2))
print('Multistart:',len(near),'of',len(runs),'reach same best residual')

"""Read saved evidence only; compare user-proposed geometric constraints.

Fit variants use only the saved known-world checkerboard image. Slider reference
positions are used solely for evaluation, never for fitting any parameter.
"""
from pathlib import Path
import copy,csv,hashlib,json,sys
import cv2,numpy as np
from scipy.optimize import least_squares
cv2.setNumThreads(1)
out=Path(__file__).resolve().parent;prev=out.parent/'new_extrinsics_v3_photo_test_20261117'
sys.path.insert(0,str(out.parents[1]/'arm_real_control'))
from vision_tools.pixel_to_world import convert_detections
raw=json.loads((prev/'provided_camera_world_sensor_raw.json').read_text())
ext=json.loads((prev/'provided_extrinsics.json').read_text())
K=np.array(raw['K']);D=np.array(raw['D']);C0=np.array(raw['C_world_mm']);R0=np.array(raw['R_world_from_camera'])
S=np.diag([-1.,-1.,1.]);Cfixed=np.array([C0[0],97.,-102.8]);target=np.array([-1.,0.,0.])
obj=np.zeros((48,3),float);obj[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)*6
m=ext['board_world_measurement'];O,U,V=[np.array(m[k],float) for k in ['origin','u_end','v_end']]
a=(U-O)/42;b=(V-O)/30;B=np.column_stack((a,b,-np.cross(a,b)))
world=O+obj@B.T;pixels=np.array(ext['candidates'][0]['ordered_pixels'])
def project(C,R):
 pc=(world-C)@R
 assert pc[:,2].min()>0
 return cv2.projectPoints(pc,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
def errors(C,R):return (project(C,R)-pixels).ravel()
def rot(v):return cv2.Rodrigues(np.asarray(v,dtype=float))[0]
def skew(v):
 x,y,z=v
 return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
# Smallest whole-frame proper rotation aligning the optical axis to -world X.
# It preserves roll as far as possible; no assumption of zero roll is added.
axis=R0[:,2];v=np.cross(axis,target);cosine=float(axis@target);A=skew(v)
Q=np.eye(3)+A+(A@A)/(1+cosine);Raxis=Q@R0
assert np.max(np.abs(Raxis[:,2]-target))<1e-12
variants=[]
def add(key,C,R,note,fit=None):
 assert np.max(np.abs(R.T@R-np.eye(3)))<1e-10 and abs(np.linalg.det(R)+1)<1e-10
 e=errors(C,R).reshape(-1,2);norm=np.linalg.norm(e,axis=1)
 cal=copy.deepcopy(raw);cal.update(name='offline_constraint_'+key,C_world_mm=C.tolist(),R_world_from_camera=R.tolist(),validation_status='offline_diagnostic_only_not_applied')
 (out/(key+'_sensor_raw.json')).write_text(json.dumps(cal,ensure_ascii=False,indent=2)+'\n')
 ku=K.copy();ku[0,2]=1919-K[0,2];ku[1,2]=1079-K[1,2];du=D.copy();du[2:4]*=-1
 upright=dict(cal,K=ku.tolist(),D=du.tolist(),R_world_from_camera=(R@S).tolist(),pixel_coordinate_frame='scene_upright')
 variants.append(dict(key=key,C_world_mm=C.tolist(),R_world_from_camera=R.tolist(),optical_axis_world=R[:,2].tolist(),
  tilt_from_minus_X_deg=float(np.degrees(np.arccos(np.clip(-R[0,2],-1,1)))),
  checkerboard_reprojection_rms_px=float(np.sqrt(np.mean(norm**2))),checkerboard_max_error_px=float(norm.max()),
  note=note,fit=fit,_cal_upright=upright))
add('A_original',C0,R0,'原候选0，完整不变')
add('B_fix_YZ_keep_X_R',Cfixed,R0,'直接固定Cy=97/Cz=-102.8，保留原Cx与全部R；没有重拟合')
add('C_axis_minus_X_keep_C',C0,Raxis,'只将光轴最小旋转至-X，保留原C；尽量保持原绕轴方向，无重拟合')
add('D_fix_YZ_axis_minus_X',Cfixed,Raxis,'固定Y/Z并将光轴最小旋转至-X，保留原Cx；无重拟合')
# Correctly re-estimate coupled parameters from checkerboard observations only.
def residual_free_R(p):return errors(np.array([p[0],97.,-102.8]),rot(p[1:])@R0)
fit1=least_squares(residual_free_R,np.array([C0[0],0.,0.,0.]),xtol=1e-13,ftol=1e-13,gtol=1e-10,max_nfev=2000)
assert fit1.success
add('E_refit_X_R_fixed_YZ',np.array([fit1.x[0],97.,-102.8]),rot(fit1.x[1:])@R0,
 '固定已知Y/Z，仅用棋盘重估Cx与三维姿态；滑杆参考不参与拟合',dict(success=bool(fit1.success),parameters=['Cx','rotation_delta_x_rad','rotation_delta_y_rad','rotation_delta_z_rad'],values=fit1.x.tolist(),nfev=fit1.nfev))
def residual_fixed_axis(p):return errors(np.array([p[0],97.,-102.8]),rot(target*p[1])@Raxis)
fit2=least_squares(residual_fixed_axis,np.array([C0[0],0.]),xtol=1e-13,ftol=1e-13,gtol=1e-10,max_nfev=2000)
assert fit2.success
add('F_refit_X_roll_fixed_YZ_axis',np.array([fit2.x[0],97.,-102.8]),rot(target*fit2.x[1])@Raxis,
 '固定Y/Z与光轴-X，仅用棋盘重估Cx及绕光轴滚转；不假定图像完全水平',dict(success=bool(fit2.success),parameters=['Cx','roll_delta_rad'],values=fit2.x.tolist(),nfev=fit2.nfev))
records=[];allrows=[]
def stats(x):
 x=np.array(x)
 return dict(n=len(x),mean_error_mm=float(x.mean()),mae_mm=float(abs(x).mean()),rmse_mm=float(np.sqrt(np.mean(x*x))),max_abs_error_mm=float(abs(x).max()),within_0_5_mm=int(np.count_nonzero(abs(x)<=.5)))
for variant in variants:
 per=[];new_errors=[]
 for photo in ['0001','0002','0005']:
  record=json.loads((prev/f'result_{photo}.json').read_text())
  converted=convert_detections(record['pixels'],variant['_cal_upright'])
  for color in ['blue','yellow']:
   value=converted['sliders'][color]['world_xyz_mm'];ref=record['sliders'][color]['reference_world_x_mm'];err=value[0]-ref
   if variant['key']=='A_original':assert abs(value[0]-record['sliders'][color]['detected_world_x_mm'])<1e-10
   item=dict(variant=variant['key'],photo=photo,color=record['sliders'][color]['color'],reference_world_x_mm=ref,world_x_mm=value[0],world_y_mm=value[1],world_z_mm=value[2],error_x_mm=err)
   per.append(item);allrows.append(item);new_errors.append(err)
 variant['photo_X_metrics']=stats(new_errors);variant['photo_results']=per;del variant['_cal_upright']
decomposition=[]
for photo in ['0001','0002','0005']:
 record=json.loads((prev/f'result_{photo}.json').read_text())
 def raw_to_upright(c):
  c=copy.deepcopy(c);k=np.array(c['K']);d=np.array(c['D']);r=np.array(c['R_world_from_camera'])
  k[0,2]=1919-k[0,2];k[1,2]=1079-k[1,2];d[2:4]*=-1
  c.update(K=k.tolist(),D=d.tolist(),R_world_from_camera=(r@S).tolist(),pixel_coordinate_frame='scene_upright')
  return c
 onlyY=copy.deepcopy(raw);onlyY['C_world_mm'][1]=97.
 onlyZ=copy.deepcopy(raw);onlyZ['C_world_mm'][2]=-102.8
 yw=convert_detections(record['pixels'],raw_to_upright(onlyY));zw=convert_detections(record['pixels'],raw_to_upright(onlyZ))
 for color in ['blue','yellow']:
  base=record['sliders'][color]['detected_world_x_mm'];dy=yw['sliders'][color]['world_xyz_mm'][0]-base;dz=zw['sliders'][color]['world_xyz_mm'][0]-base
  assert abs(dz)<1e-12
  decomposition.append(dict(photo=photo,color=color,X_change_from_Cy_only_mm=dy,X_change_from_Cz_only_mm=dz))
summary=dict(user_proposed_Cy_Cz_mm=[97.,-102.8],Cz_is_latest_proposed_value_not_prior_104_22=True,original_tilt_deg=variants[0]['tilt_from_minus_X_deg'],
 original_optical_axis_world=axis.tolist(),center_C_change_mm=(Cfixed-C0).tolist(),
 original_axis_Z_drift_for_100mm_optical_depth_mm=float(100*axis[2]/(-axis[0])),
 variants=variants,center_change_decomposition=decomposition,photo_references_used_for_fitting=False,checkerboard_world_pose_independently_verified=False,
 excluded_photos=['0003','0004'],new_image_detection_performed=False,same_prior_pixels_and_X_references=True,
 production_or_board_modified=False,live_camera_or_UART_opened=False,
 source_sha256={str(p.relative_to(out.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [prev/'provided_camera_world_sensor_raw.json',prev/'provided_extrinsics.json']+[prev/f'result_{p}.json' for p in ['0001','0002','0005']]})
(out/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (out/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
print(json.dumps([{k:v for k,v in var.items() if k not in ['R_world_from_camera','photo_results','fit']} for var in variants],ensure_ascii=False,indent=2))

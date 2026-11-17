"""Portable offline reproduction of this archive; no camera/UART access."""
from pathlib import Path
import copy,hashlib,json,math,sys
import cv2,numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'arm_real_control'))
from vision_tools.detect_sliders import detect_frame
from vision_tools.pixel_to_world import convert_detections
cv2.setNumThreads(1)
prev=ROOT/'diagnostics/new_extrinsics_v3_photo_test_20261117'
raw=json.loads((prev/'provided_camera_world_sensor_raw.json').read_text())
ext=json.loads((prev/'provided_extrinsics.json').read_text())
published=json.loads((ROOT/'diagnostics/strict_mount_extrinsics_20261117/solution_report.json').read_text())
K=np.array(raw['K']);D=np.array(raw['D'])
obj=np.zeros((48,3),float);obj[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)*6
O,U,V=[np.array(ext['board_world_measurement'][k],float) for k in ['origin','u_end','v_end']]
a=(U-O)/42;b=(V-O)/30;world=O+obj@np.column_stack([a,b,-np.cross(a,b)]).T
observed=np.array(ext['candidates'][0]['ordered_pixels'])
def rotation(phi):
 c,s=math.cos(phi),math.sin(phi)
 return np.array([[0.,0.,-1.],[c,-s,0.],[s,c,0.]])
def project(p):
 pc=(world-[p[0],97.,-102.8])@rotation(p[1])
 return cv2.projectPoints(pc,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
fit=least_squares(lambda p:(project(p)-observed).ravel(),[236.63,0.],bounds=([O[0]+.1,-np.inf],[2000.,np.inf]),xtol=1e-13,ftol=1e-13,gtol=1e-10,max_nfev=2000)
assert fit.success
residual=np.linalg.norm(project(fit.x)-observed,axis=1);rms=float(np.sqrt(np.mean(residual**2)))
assert abs(rms-published['strict_solution']['checkerboard_RMS_px'])<1e-7
assert abs(fit.x[0]-published['strict_solution']['C_world_mm'][0])<1e-4
strict=copy.deepcopy(raw);strict.update(C_world_mm=[float(fit.x[0]),97.,-102.8],R_world_from_camera=rotation(fit.x[1]).tolist())
def upright(cal):
 c=copy.deepcopy(cal);k=np.array(c['K']);d=np.array(c['D']);k[0,2]=1919-k[0,2];k[1,2]=1079-k[1,2];d[2:4]*=-1
 c.update(K=k.tolist(),D=d.tolist(),R_world_from_camera=(np.array(c['R_world_from_camera'])@np.diag([-1.,-1.,1.])).tolist(),pixel_coordinate_frame='scene_upright')
 return c
rows=[]
for src in json.loads((prev/'input_manifest.json').read_text()):
 photo=src['photo'];assert photo in ['0001','0002','0005']
 path=ROOT/'diagnostics/photo_redetect_76459ec8_20261117/input/session_20260510_222231_76459ec8'/Path(src['original_absolute_path']).name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==src['sha256']
 pixels=detect_frame(cv2.imread(str(path)),profile='real',already_rotated=False)
 original=json.loads((prev/f'result_{photo}.json').read_text());assert pixels==original['pixels']
 for name,cal in [('original',raw),('strict',strict)]:
  converted=convert_detections(pixels,upright(cal))
  for color in ['blue','yellow']:
   value=converted['sliders'][color]['world_xyz_mm'][0];ref=original['sliders'][color]['reference_world_x_mm']
   rows.append(dict(photo=photo,color=color,variant=name,reference_X_mm=ref,detected_X_mm=value,error_X_mm=value-ref))
metrics={}
for name in ['original','strict']:
 e=np.array([r['error_X_mm'] for r in rows if r['variant']==name])
 metrics[name]=dict(MAE_mm=float(abs(e).mean()),max_abs_error_mm=float(abs(e).max()))
assert abs(metrics['original']['MAE_mm']-1.9802179475101813)<1e-8
assert abs(metrics['strict']['MAE_mm']-published['photo_metrics']['MAE_mm'])<1e-4
result=dict(C_world_mm=strict['C_world_mm'],R_world_from_camera=strict['R_world_from_camera'],checkerboard_RMS_px=rms,
 metrics=metrics,results=rows,archived_results_reproduced=True,live_camera_or_UART_opened=False,validation_passed=False)
out=ROOT/'reproduction_results';out.mkdir(exist_ok=True)
(out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False,indent=2))

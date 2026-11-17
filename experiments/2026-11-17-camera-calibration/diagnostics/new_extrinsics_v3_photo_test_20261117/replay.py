"""Offline replay of only user-approved ground photos 1,2,5; no camera access."""
from pathlib import Path
import copy,csv,hashlib,json,shutil,subprocess,sys
import cv2,numpy as np
OUT=Path(__file__).resolve().parent
WORKSPACE=OUT.parents[1];REPO=WORKSPACE/'arm_real_control'
PREVIOUS=WORKSPACE/'diagnostics/photo_redetect_76459ec8_20261117'
sys.path.insert(0,str(REPO))
from vision_tools.detect_sliders import detect_frame
from vision_tools.pixel_to_world import convert_detections
from control.vision import Vision
cv2.setNumThreads(1)
def save(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw_path=OUT/'provided_camera_world_sensor_raw.json';intr_path=OUT/'provided_intrinsics.json'
raw=json.loads(raw_path.read_text());intr=json.loads(intr_path.read_text());old=json.loads((WORKSPACE/'diagnostics/new_calibration_photo_test_20261117/used_calibration_scene_upright.json').read_text())
review=json.loads((OUT/'provided_extrinsics.json').read_text())
assert raw==review['candidates'][0]['calibration_sensor_raw'], 'Not the reviewed candidate 0'
assert raw['pixel_coordinate_frame']=='sensor_raw' and raw['image_size_px']==[1920,1080]
assert np.array_equal(raw['K'],intr['camera_matrix']) and np.array_equal(raw['D'],intr['distortion_coefficients'])
def browser_integer_normalize(x):
 if isinstance(x,float) and x.is_integer():return int(x)
 if isinstance(x,list):return [browser_integer_normalize(v) for v in x]
 if isinstance(x,dict):return {k:browser_integer_normalize(v) for k,v in x.items()}
 return x
canonical_sha=lambda d:hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()
normalized_source=browser_integer_normalize(intr)
assert normalized_source==intr
assert raw['intrinsics_source_sha256']==review['intrinsics']['source_sha256']==canonical_sha(normalized_source)
save('intrinsic_source_normalization_check.json',dict(uploaded_canonical_sha256=canonical_sha(intr),browser_integer_normalized_sha256=canonical_sha(normalized_source),referenced_source_sha256=raw['intrinsics_source_sha256'],numerically_identical=True,explanation='Browser JSON serialization changes integral floats such as 6.0 and 0.0 to integers. Exact referenced hash reproduced without any numeric value change.'))
assert raw['intrinsics_device']=='/dev/video31' and raw['world_handedness']=='left'
(OUT/'provided_camera_world_sensor_raw.json').write_bytes(raw_path.read_bytes())
(OUT/'provided_intrinsics.json').write_bytes(intr_path.read_bytes())
cal=copy.deepcopy(raw);K=np.array(raw['K']);D=np.array(raw['D']);R=np.array(raw['R_world_from_camera']);S=np.diag([-1.,-1.,1.])
Ku=K.copy();Ku[0,2]=1919-K[0,2];Ku[1,2]=1079-K[1,2];Du=D.copy();Du[2:4]*=-1
cal.update(K=Ku.tolist(),D=Du.tolist(),R_world_from_camera=(R@S).tolist(),pixel_coordinate_frame='scene_upright')
save('used_calibration_scene_upright.json',cal)
save('conversion_receipt.json',dict(source_sha256=sha(raw_path),intrinsics_file_sha256=sha(intr_path),reviewed_candidate=0,
 physical_camera_pose_unchanged_by_conversion=True,image_rotation_deg=180,formula_R='R_upright=R_raw@diag(-1,-1,1)',
 formula_K='cx_upright=1919-cx_raw;cy_upright=1079-cy_raw;fx/fy unchanged',formula_D='p1/p2 change sign; radial unchanged',
 actual_calibration_C=cal['C_world_mm'],fit_or_compensation_performed=False))
def independent_brown_inverse(sensor_pixel):
 target=np.linalg.solve(K,np.r_[sensor_pixel,1.])[:2]
 x,y=target.copy();k1,k2,p1,p2,k3=D
 for iteration in range(30):
  r2=x*x+y*y;rad=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
  f=np.array([x*rad+2*p1*x*y+p2*(r2+2*x*x),y*rad+p1*(r2+2*y*y)+2*p2*x*y])-target
  if np.linalg.norm(f)<1e-14:return np.array([x,y])
  slope=k1+2*k2*r2+3*k3*r2*r2;rx=2*x*slope;ry=2*y*slope
  J=np.array([[rad+x*rx+2*p1*y+6*p2*x,x*ry+2*p1*x+2*p2*y],
              [y*rx+2*p1*x+2*p2*y,rad+y*ry+6*p1*y+2*p2*x]])
  step=np.linalg.solve(J,f);x-=step[0];y-=step[1]
 raise ValueError('Independent Brown inverse did not converge')

source_files=['vision_tools/detect_sliders.py','vision_tools/pixel_to_world.py','control/vision.py']
hashes={f:sha(REPO/f) for f in source_files}
for f in source_files:
 p=OUT/'used_source'/f;p.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(REPO/f,p)
config=copy.deepcopy(json.loads((PREVIOUS/'used_vision_config.json').read_text()));config['calibration']=cal
save('used_vision_config.json',config)
vision=Vision(REPO,config);excluded=json.loads((PREVIOUS/'exclusions.json').read_text());save('exclusions.json',excluded)
manifest=[];rows=[];results=[]
for photo in excluded['included_photos']:
 assert photo not in excluded['excluded_photos']
 prior=json.loads((PREVIOUS/('result_'+photo+'.json')).read_text());name=prior['image']
 image_path=next((PREVIOUS/'input').rglob(name));meta_path=image_path.with_suffix('.json');meta=json.loads(meta_path.read_text())
 assert sha(image_path)==meta['sha256_png']==prior['source_image_sha256']
 assert meta['pixel_coordinate_frame']=='sensor_raw' and meta['saved_image_rotation_deg']==0
 image=cv2.imread(str(image_path));assert list(image.shape[1::-1])==[1920,1080]
 pixels=detect_frame(image,profile='real',already_rotated=False)
 same_pixels=json.dumps(pixels,sort_keys=True)==json.dumps(prior['pixels'],sort_keys=True)
 assert same_pixels,'Pixel detector changed versus prior replay'
 upright=cv2.rotate(image,cv2.ROTATE_180)
 assert pixels==detect_frame(upright,profile='real',already_rotated=True)
 world=convert_detections(pixels,cal);baseline=convert_detections(pixels,old)
 assert world['calibration']['C_world_mm']==cal['C_world_mm'] and world['calibration']['R_world_from_camera']==cal['R_world_from_camera']
 obs=vision._detect(None,lambda *a,**k:pixels,'offline_new_calibration_replay')
 record=dict(photo=photo,image=name,source_image_sha256=sha(image_path),reference_note=meta['note'],
  input_frame='sensor_raw',image_rotated_once=True,pixels_identical_to_previous=True,
  metadata_old_reference_geometry_ignored=True,pixels=pixels,world=world,task_observation=obs,sliders={})
 for color,zh in [('blue','蓝'),('yellow','橙')]:
  pixel=np.array(pixels['sliders'][color]['pixel_xy'],dtype=float);sensor_pixel=np.array([1919.,1079.])-pixel
  normalized=independent_brown_inverse(sensor_pixel);ray=R@np.r_[normalized,1.]
  C=np.array(raw['C_world_mm']);independent=C+(raw['plane_y_mm']-C[1])/ray[1]*ray
  xyz=np.array(world['sliders'][color]['world_xyz_mm']);diff=float(np.max(np.abs(xyz-independent)));assert diff<1e-7, dict(diff=diff, xyz=xyz.tolist(), independent=independent.tolist(), pixel=pixel.tolist(), sensor_pixel=sensor_pixel.tolist(),cal=world['calibration'])
  reference=prior['reference_world_x_mm'][color];before=baseline['sliders'][color]['world_xyz_mm'][0]
  v2record=json.loads((WORKSPACE/'diagnostics/new_calibration_photo_test_20261117'/('result_'+photo+'.json')).read_text())
  assert abs(before-v2record['sliders'][color]['detected_world_x_mm'])<1e-9
  item=dict(color=zh,reference_world_x_mm=reference,detected_world_x_mm=float(xyz[0]),error_world_x_mm=float(xyz[0]-reference),
   world_xyz_mm=xyz.tolist(),arm_xyz_mm=[65.-xyz[0],xyz[1]-26.5,xyz[2]+64.5],
   previous_world_x_mm=before,previous_error_world_x_mm=before-reference,pixel_xy_upright=pixel.tolist(),pixel_xy_sensor_raw=sensor_pixel.tolist(),
   independent_sensor_raw_world_xyz_mm=independent.tolist(),independent_ray_difference_mm=diff,
   detector_reliable=pixels['sliders'][color]['reliable'],task_reliable=obs['reliability'][zh]['reliable'],task_reasons=obs['reliability'][zh]['reasons'])
  record['sliders'][color]=item
  rows.append(dict(photo=photo,color=zh,reference_world_x_mm=reference,previous_world_x_mm=before,
   previous_error_mm=before-reference,new_world_x_mm=float(xyz[0]),new_error_mm=float(xyz[0]-reference),
   new_arm_x_mm=float(65.-xyz[0]),new_world_z_mm=float(xyz[2]),detector_reliable=item['detector_reliable'],task_reliable=item['task_reliable'],task_reasons=';'.join(item['task_reasons'])))
 results.append(record);save('result_'+photo+'.json',record)
 # Overlay measured rail/shaft points only, no modification of raw evidence.
 canvas=upright.copy();ep=pixels['rail_endpoints_px']
 if ep:
  cv2.line(canvas,tuple(np.rint(ep[0]).astype(int)),tuple(np.rint(ep[1]).astype(int)),(0,255,0),2,cv2.LINE_AA)
 for color,bgr,offset in [('blue',(255,150,30),-60),('yellow',(0,210,255),45)]:
  item=record['sliders'][color];p=tuple(np.rint(item['pixel_xy_upright']).astype(int));cv2.drawMarker(canvas,p,bgr,cv2.MARKER_CROSS,24,3)
  label=f'{color}: X={item["detected_world_x_mm"]:.2f} mm; error={item["error_world_x_mm"]:+.2f}'
  pos=(max(10,p[0]-180),p[1]+offset)
  cv2.putText(canvas,label,pos,cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA);cv2.putText(canvas,label,pos,cv2.FONT_HERSHEY_SIMPLEX,.65,bgr,2,cv2.LINE_AA)
 panel=np.zeros((170,1920,3),np.uint8)
 lines=[f'Photo {photo} | New video31 K/D + reviewed pose candidate 0 | 180-degree rotation once',
  'Blue: ref %.2f | old %.3f (%+.3f) | new %.3f (%+.3f) mm' % tuple(record['sliders']['blue'][k] for k in ['reference_world_x_mm','previous_world_x_mm','previous_error_world_x_mm','detected_world_x_mm','error_world_x_mm']),
  'Orange: ref %.2f | old %.3f (%+.3f) | new %.3f (%+.3f) mm' % tuple(record['sliders']['yellow'][k] for k in ['reference_world_x_mm','previous_world_x_mm','previous_error_world_x_mm','detected_world_x_mm','error_world_x_mm'])]
 for i,line in enumerate(lines):cv2.putText(panel,line,(20,37+i*48),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),1,cv2.LINE_AA)
 cv2.imwrite(str(OUT/('detected_'+photo+'.jpg')),np.vstack([canvas,panel]),[cv2.IMWRITE_JPEG_QUALITY,94])
 manifest.append(dict(photo=photo,original_absolute_path=str(image_path),metadata_path=str(meta_path),sha256=sha(image_path),bytes=image_path.stat().st_size,reference_note=meta['note']))
with (OUT/'results.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def stats(values):
 a=np.array(values);return dict(n=len(a),mean_error_mm=float(a.mean()),mae_mm=float(np.abs(a).mean()),rmse_mm=float(np.sqrt(np.mean(a*a))),max_abs_error_mm=float(np.abs(a).max()),within_0_5_mm=int(np.count_nonzero(np.abs(a)<=.5)))
metrics={}
for group in ['蓝','橙','combined']:
 data=rows if group=='combined' else [x for x in rows if x['color']==group]
 metrics[group]=dict(previous=stats([r['previous_error_mm'] for r in data]),new=stats([r['new_error_mm'] for r in data]))
summary=dict(included_photos=excluded['included_photos'],excluded_photos=excluded['excluded_photos'],images=len(results),points=len(rows),
 previous_calibration='Previous video31 30-photo intrinsics plus prior extrinsics candidate 0, same pixels and world references',
 new_calibration='Provided video31 34-photo intrinsics plus new extrinsics candidate 0, transformed consistently into scene_upright',
 all_pixels_identical_to_previous=True,all_180_rotation_checks_pass=True,independent_raw_ray_verification_max_mm=max(x['independent_ray_difference_mm'] for r in results for x in r['sliders'].values()),
 all_detector_pairs_reliable=sum(all(i['detector_reliable'] for i in r['sliders'].values()) for r in results),
 all_task_pairs_reliable=sum(all(i['task_reliable'] for i in r['sliders'].values()) for r in results),
 git_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),source_sha256=hashes,metrics=metrics,
 parameter_fitting=False,production_or_board_modified=False,live_camera_opened=False,UART_opened=False,
 old_photo_replay_assumes_same_camera_lens_focus_and_mount=True,prior_user_confirmation='镜头、焦点和相机位置都未改变',no_new_physical_pose_measurement_performed=True,
 reference_precision_note='Manual metadata world X references; measurement uncertainty not supplied')
for name,h in hashes.items():assert sha(REPO/name)==h
save('input_manifest.json',manifest);save('summary.json',summary)
print(json.dumps(dict(rows=rows,summary=summary),ensure_ascii=False,indent=2))

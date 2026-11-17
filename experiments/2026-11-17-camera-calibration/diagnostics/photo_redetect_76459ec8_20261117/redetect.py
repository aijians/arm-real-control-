"""Offline detection on selected new PNGs; respect the user's photo exclusions."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
EXCLUSIONS = json.loads((OUT/'exclusions.json').read_text())
WORKSPACE = OUT.parents[1]
REPO = WORKSPACE/'arm_real_control'
sys.path.insert(0,str(REPO))
from control.vision import Vision
from vision_tools.detect_sliders import detect_frame
from vision_tools.pixel_to_world import convert_detections

cv2.setNumThreads(1)
cal = json.loads((REPO/'config/camera_world_20261117.json').read_text())
source_names = ['control/vision.py','vision_tools/pixel_to_world.py','vision_tools/detect_sliders.py']
source_hashes = {n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in source_names}
for name in source_names:
    target=OUT/'used_source'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(REPO/name,target)
(OUT/'used_calibration.json').write_text(json.dumps(cal,ensure_ascii=False,indent=2)+'\n')
config = copy.deepcopy(json.loads((REPO/'config/robot.json').read_text())['vision'])
config.pop('shoulder_global_xy_mm',None)
config.update(profile='real',already_rotated=False,calibration=cal,
              shoulder_global_xyz_mm=[65.,26.5,-64.5],world_to_model_axis_signs=[-1,1,1],
              fixed_slider_limits_x_mm={'蓝':[-48.5,-4.],'橙':[-33.,11.5]})
(OUT/'used_vision_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
vision = Vision(REPO,config)
results=[]
table=[]
previews=[]
for meta_path in sorted((OUT/'input').rglob('photo_*.json')):
    if meta_path.stem.split('_')[1] in EXCLUSIONS['excluded_photos']:
        continue
    meta=json.loads(meta_path.read_text())
    image_path=meta_path.parent/meta['image']
    data=image_path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==meta['sha256_png'],image_path
    image=cv2.imread(str(image_path))
    assert image is not None and list(image.shape[1::-1])==meta['image_size_px']
    assert meta['pixel_coordinate_frame']=='sensor_raw' and meta['saved_image_rotation_deg']==0
    match=re.search(r'橙\s*[:：]\s*([\d.]+).*蓝\s*[;；:：]\s*([\d.]+)',meta['note'])
    assert match,meta['note']
    refs={'blue':float(match.group(2)),'yellow':float(match.group(1))}
    pixels=detect_frame(image,profile='real',already_rotated=False)
    upright=cv2.rotate(image,cv2.ROTATE_180)
    manual=detect_frame(upright,profile='real',already_rotated=True)
    assert pixels==manual,'180 degree mismatch: '+meta['image']
    ground=detect_frame(image,profile='ground',already_rotated=False)
    world=convert_detections(pixels,cal)
    old_world=convert_detections(pixels)
    observation=vision._detect(None,lambda *a,**kw:pixels,'offline')
    row=dict(image=meta['image'],reference_note=meta['note'],
             source_image_sha256=meta['sha256_png'],rotation_equivalence=True,
             ground_real_pixel_identical=all(ground[k]==pixels[k] for k in
                                            ('sliders','rail_line','rail_endpoints_px')),
             reference_source='User-entered world X values in per-image metadata notes; not auto measured.',
             metadata_reference_geometry_overridden=True,pixels=pixels,world=world,
             task_observation=observation,reference_world_x_mm=refs,sliders={})
    compact=dict(photo=meta_path.stem.split('_')[1],image=meta['image'],
                 rail_k=pixels['rail_line']['k'],rail_m=pixels['rail_line']['m'],
                 rail_source=pixels['rail_line']['source'])
    for color,zh in [('blue','蓝'),('yellow','橙')]:
        pix=pixels['sliders'][color]
        wx=world['sliders'][color]['world_xyz_mm'][0]
        ref=refs[color]
        # For a fair old/new comparison, express the old arm-local X in the
        # same new reporting world; do not compare opposite-axis global values.
        old_arm_x=old_world['sliders'][color]['world_xyz_mm'][0]-189.3
        old_as_new_world_x=65.-old_arm_x
        item=dict(reference_world_x_mm=ref,detected_world_x_mm=wx,error_world_x_mm=wx-ref,
                  reference_arm_x_mm=65.-ref,detected_arm_x_mm=observation['sliders_x_mm'][zh],
                  old_prediction_expressed_in_new_world_x_mm=old_as_new_world_x,
                  old_error_against_same_reference_mm=old_as_new_world_x-ref,
                  pixel_xy=pix['pixel_xy'],detector_reliable=pix['reliable'],
                  task_reliable=observation['reliability'][zh]['reliable'],
                  task_reasons=observation['reliability'][zh]['reasons'])
        row['sliders'][color]=item
        label='blue' if color=='blue' else 'orange'
        compact.update({label+'_reference_world_x_mm':ref,label+'_detected_world_x_mm':wx,
                        label+'_error_mm':wx-ref,label+'_arm_x_mm':item['detected_arm_x_mm'],
                        label+'_old_error_mm':item['old_error_against_same_reference_mm'],
                        label+'_detector_reliable':item['detector_reliable'],
                        label+'_task_reliable':item['task_reliable'],
                        label+'_task_reasons':';'.join(item['task_reasons'])})
    results.append(row)
    table.append(compact)
    canvas=upright.copy()
    ep=pixels.get('rail_endpoints_px')
    if ep:
        cv2.line(canvas,tuple(np.rint(ep[0]).astype(int)),tuple(np.rint(ep[1]).astype(int)),(0,255,0),2,cv2.LINE_AA)
        for pt in ep:cv2.circle(canvas,tuple(np.rint(pt).astype(int)),6,(0,255,0),2)
    for color,bgr,offset in [('blue',(255,150,30),-55),('yellow',(0,200,255),40)]:
        pt=tuple(np.rint(row['sliders'][color]['pixel_xy']).astype(int))
        cv2.drawMarker(canvas,pt,bgr,cv2.MARKER_CROSS,24,3)
        label=f'{color}: Xw={row["sliders"][color]["detected_world_x_mm"]:.2f} mm'
        cv2.putText(canvas,label,(pt[0]-110,pt[1]+offset),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(canvas,label,(pt[0]-110,pt[1]+offset),cv2.FONT_HERSHEY_SIMPLEX,.65,bgr,2,cv2.LINE_AA)
    panel=np.zeros((195,1920,3),dtype=np.uint8)
    photo=compact['photo']
    lines=[f'Photo {photo} | video31 | rotated 180 once | new measured geometry, original borrowed K/D',
           f'Blue   reference={refs["blue"]:.2f}   detected={compact["blue_detected_world_x_mm"]:.2f}   error={compact["blue_error_mm"]:+.2f} mm',
           f'Orange reference={refs["yellow"]:.2f}   detected={compact["orange_detected_world_x_mm"]:.2f}   error={compact["orange_error_mm"]:+.2f} mm',
           f'Pixel reliable: blue={compact["blue_detector_reliable"]}, orange={compact["orange_detector_reliable"]} | No parameter fitting']
    for i,line in enumerate(lines):
        cv2.putText(panel,line,(20,36+i*46),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),1,cv2.LINE_AA)
    preview=np.vstack([canvas,panel])
    path=OUT/('detected_'+photo+'.jpg')
    cv2.imwrite(str(path),preview,[cv2.IMWRITE_JPEG_QUALITY,93])
    previews.append(cv2.resize(preview,(960,638)))
    (OUT/('result_'+photo+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')

with (OUT/'results.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(table[0]))
    writer.writeheader();writer.writerows(table)

def error_stats(data):
    a=np.asarray(data,dtype=float)
    return dict(n=len(a),mean_error_mm=float(a.mean()),mae_mm=float(np.abs(a).mean()),
                rmse_mm=float(np.sqrt(np.mean(a*a))),min_error_mm=float(a.min()),
                max_error_mm=float(a.max()),max_abs_error_mm=float(np.abs(a).max()))

stats={}
for color,label in [('blue','blue'),('yellow','orange')]:
    stats[label]=dict(new=error_stats([r['sliders'][color]['error_world_x_mm'] for r in results]),
                     old_same_reference=error_stats([r['sliders'][color]['old_error_against_same_reference_mm'] for r in results]),
                     detector_reliable=sum(r['sliders'][color]['detector_reliable'] for r in results),
                     task_reliable=sum(r['sliders'][color]['task_reliable'] for r in results))
stats['combined']=error_stats([r['sliders'][c]['error_world_x_mm'] for r in results for c in ('blue','yellow')])
summary=dict(images=len(results),samples=2*len(results),all_input_png_hashes_verified=True,
             included_photos=EXCLUSIONS['included_photos'],excluded_photos=EXCLUSIONS['excluded_photos'],
             all_rotation_equivalence=all(r['rotation_equivalence'] for r in results),
             ground_real_pixels_identical=all(r['ground_real_pixel_identical'] for r in results),
             all_detector_pairs_reliable=sum(all(x['detector_reliable'] for x in r['sliders'].values()) for r in results),
             all_task_pairs_reliable=sum(all(x['task_reliable'] for x in r['sliders'].values()) for r in results),
             git_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
             source_hashes=source_hashes,used_camera_C=cal['C_world_mm'],used_shoulder=[65.,26.5,-64.5],
             original_metadata_geometry_used=False,stats=stats,
             parameter_fit_performed=False,production_modified=False,live_camera_opened=False,UART_opened=False,
             reference_precision_note='Errors are relative to the manually entered per-image X values; their measurement uncertainty is not included.')
for name,sha in source_hashes.items():
    assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==sha,'Source changed during analysis: '+name
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
blank=np.zeros_like(previews[0])
cv2.putText(blank,'Reference = user-entered world X (mm)',(35,100),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),1)
cv2.putText(blank,'C=(228.99,97,-104.22); S=(65,26.5,-64.5)',(35,150),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
cv2.putText(blank,'Fixed orthogonal basis; no calibration fitting',(35,200),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
tiles=previews+[blank]
if len(tiles)%2:tiles.append(blank)
cv2.imwrite(str(OUT/'detection_overview.jpg'),np.vstack([
    np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]),[cv2.IMWRITE_JPEG_QUALITY,93])
print(json.dumps({'table':table,'summary':summary},ensure_ascii=False,indent=2))

from pathlib import Path
import hashlib
import json
import zipfile

import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
summary=json.loads((OUT/'summary.json').read_text())
excluded=json.loads((OUT/'exclusions.json').read_text())
results=[json.loads((OUT/f'result_{n}.json').read_text()) for n in excluded['included_photos']]
assert len(results)==3 and excluded['excluded_photos']==['0003','0004']
panels=[]
for r in results:
    photo=r['image'].split('_')[1]
    image_path=next((OUT/'input').rglob(r['image']))
    image=cv2.rotate(cv2.imread(str(image_path)),cv2.ROTATE_180)
    x0,y0,x1,y1=400,140,740,500
    crop=image[y0:y1,x0:x1].copy()
    ep=r['pixels']['rail_endpoints_px']
    if ep:
        points=[tuple(np.rint(np.array(p)-[x0,y0]).astype(int)) for p in ep]
        cv2.line(crop,*points,(0,255,0),1,cv2.LINE_AA)
    for c,bgr in [('blue',(255,130,0)),('yellow',(0,210,255))]:
        point=tuple(np.rint(np.array(r['sliders'][c]['pixel_xy'])-[x0,y0]).astype(int))
        cv2.drawMarker(crop,point,bgr,cv2.MARKER_CROSS,12,1)
    crop=cv2.resize(crop,(680,720),interpolation=cv2.INTER_LINEAR)
    title=np.zeros((110,680,3),np.uint8)
    lines=[f'Photo {photo} | WORLD X, mm',
           f'Blue   ref {r["sliders"]["blue"]["reference_world_x_mm"]:.2f}, detected {r["sliders"]["blue"]["detected_world_x_mm"]:.2f}',
           f'Orange ref {r["sliders"]["yellow"]["reference_world_x_mm"]:.2f}, detected {r["sliders"]["yellow"]["detected_world_x_mm"]:.2f}']
    for i,line in enumerate(lines):cv2.putText(title,line,(12,26+34*i),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    panels.append(np.vstack([title,crop]))
blank=np.zeros_like(panels[0])
for i,line in enumerate(['New video31 PNGs; original borrowed K/D',
                          'C = (228.99, 97, -104.22) mm',
                          'S = (65, 26.5, -64.5) mm',
                          'Rotate raw images 180 once',
                          'Green: rail endpoints/reference line',
                          'Crosses: detected shaft-axis pixels',
                          'No parameter fitting or hardware changes',
                          'Photos 3 and 4 excluded by user']):
    cv2.putText(blank,line,(14,55+60*i),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
cv2.imwrite(str(OUT/'导轨检测放大对照.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack([panels[2],blank])]),[cv2.IMWRITE_JPEG_QUALITY,95])
full=[cv2.resize(cv2.imread(str(OUT/f'detected_{n}.jpg')),(960,638)) for n in excluded['included_photos']]
cv2.imwrite(str(OUT/'detection_overview.jpg'),np.vstack([np.hstack(full[:2]),np.hstack([full[2],np.zeros_like(full[0])])]),[cv2.IMWRITE_JPEG_QUALITY,93])

table=['|照片|蓝参考X|蓝识别X|蓝偏差|橙参考X|橙识别X|橙偏差|',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in results:
    b,o=r['sliders']['blue'],r['sliders']['yellow']
    name=r['image'].split('_')[1]
    table.append(f'|{name}|{b["reference_world_x_mm"]:.2f}|{b["detected_world_x_mm"]:.2f}|{b["error_world_x_mm"]:+.2f}|{o["reference_world_x_mm"]:.2f}|{o["detected_world_x_mm"]:.2f}|{o["error_world_x_mm"]:+.2f}|')
report='''# 最新实拍图重新检测：仅第1、2、5张

原包含5张1920×1080 PNG，ZIP CRC及各图与配套JSON的SHA256均通过。用户随后明确第3、4张“就不测试了，可能有误”，本正式结果只保留第1、2、5张，共6个杆位参考点；第3、4张不再检测、不进入统计、图表或交付结果。用户排除前的初步数据仅在excluded_preliminary/归档，原始ZIP及输入未改。编号日期沿用原包，采集时间来自板端时钟，不据此声称真实日历采集日期。

## 本次参数与范围

- 光心C=(228.99,97,−104.22)mm，肩轴原点S=(65,26.5,−64.5)mm；使用用户最新文字确认值。
- 新世界只有X反向，机械臂局部系不变；转正矩阵`[[0,0,-1],[-1,0,0],[0,-1,0]]`，杆轴平面世界Y=181.5mm。`X_arm=65−X_world`。
- 相机原图倒装180°，统一real入口转正一次。保留3张的原图自动转正与手动转正输入逐项检测一致，ground兼容入口像素结果亦一致。
- 内参仍用原video22标定的K/D，并按180°转正变换；这批图来自video31，尚没有该相机新棋盘标定。
- 导出JSON里的reference_geometry仍是旧C24.5/99/−108、肩XY189.3/28.5、轴心Y183.5。本轮仅保留它作为导出记录，未用于此次新世界计算。
- 各图note中的蓝/橙世界X用作用户填写的实测参考；没有把它们用于拟合C、R、K、D、经验比例或偏移，未修改原包或生产配置。

## 逐图结果

单位均为新世界坐标mm，偏差=识别−备注参考。

TABLE

3/3张双杆像素可靠，按候选固定行程和原2.5mm止挡容差，3/3张也均通过任务视觉可靠性检查。这个“可靠”只检查杆身测量、遮挡、宽度及读数不超行程等条件，不能表示毫米精度达标。相同X范围内的4～5mm误差可通过这些检查。

保留的6个点在新世界X方向全部低估，蓝杆平均偏差−5.02mm、RMSE5.02mm，橙杆平均偏差−4.26mm、RMSE4.27mm。6点整体平均偏差−4.64mm，MAE4.64mm、RMSE4.66mm，最大绝对偏差5.20mm。以上均相对备注实测值计算，未包含手工量测的不确定性。本组没有达到±0.5mm定位精度。

## 本组现象与解释边界

原始PNG中也存在与录像同量级的偏差，说明问题并非仅存在于MP4压缩回放。保留数据的两杆误差方向一致，更接近整体映射偏移；但仅凭这3张不能唯一归因于内参、相机光心、杆轴测量平面或像素算法。

相对位移的两个例子：第1→5张蓝杆参考移动−8.50mm，识别移动−8.4993mm；第1→2张橙杆参考移动+20.00mm，识别移动+20.1126mm。这说明这些样本中相对位移可接近参考，而绝对位置仍有同向偏差，不能只验证两杆间距或位移就认证绝对定位。相同蓝113.5参考的第1、2张识别读数相差0.546mm，包含手工复位/参考量测等不确定性，不等于纯算法静止抖动。

内参仍为已经确认的跨相机复用参数，需要当前video31同取流模式的棋盘标定与独立位置共同验证。本轮没有拟合C、R、K、D或给读数统一加减补偿。

## 标注图和可复核数据

绿色标示检测得到的导轨端点/参考线，蓝、黄十字为杆轴像素位置。标注图已转正，输入原图保持原样。

![导轨检测放大对照](ZOOM)

完整视场标注仅为detected_0001.jpg、detected_0002.jpg、detected_0005.jpg。每图result_*.json保留原检测像素、世界坐标、局部坐标、可靠性原因、端点诊断与参考误差；results.csv和summary.json提供汇总。used_calibration.json、used_vision_config.json及used_source/保存实际使用参数和源文件；源码commit29a9e2dd522aabd5b6943d89ee3322283f72e0e1，处理前后源码SHA不变。

本轮全部离线，不打开实时相机/UART、不运动、不改PID/offset、代码或运行配置，未部署或重新标定。
'''.replace('TABLE','\n'.join(table)).replace('ZOOM',str(OUT/'导轨检测放大对照.jpg'))
(OUT/'重新检测报告.md').write_text(report)
names=['重新检测报告.md','results.csv','summary.json','used_calibration.json','used_vision_config.json',
       '导轨检测放大对照.jpg','detection_overview.jpg','input_manifest.json','exclusions.json','redetect.py','make_report.py']
names += [f'result_{n}.json' for n in excluded['included_photos']]
names += [f'detected_{n}.jpg' for n in excluded['included_photos']]
names += [str(p.relative_to(OUT)) for p in (OUT/'used_source').rglob('*') if p.is_file()]
manifest={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in names}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(OUT/'重新检测结果.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in [*names,'manifest.json']:z.write(OUT/name,name)
with zipfile.ZipFile(OUT/'重新检测结果.zip') as z:
    assert z.testzip() is None
    assert not any('0003' in n or '0004' in n or 'excluded_preliminary' in n for n in z.namelist())
    for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
(OUT/'package_receipt.json').write_text(json.dumps({'zip':str(OUT/'重新检测结果.zip'),
    'bytes':(OUT/'重新检测结果.zip').stat().st_size,
    'sha256':hashlib.sha256((OUT/'重新检测结果.zip').read_bytes()).hexdigest(),
    'crc_pass':True,'all_packaged_sha256_verified':True,'included_photos':excluded['included_photos'],
    'excluded_photos':excluded['excluded_photos'],'original_photos_included':False,'source_zip_preserved':True},ensure_ascii=False,indent=2)+'\n')
print(OUT/'重新检测报告.md')

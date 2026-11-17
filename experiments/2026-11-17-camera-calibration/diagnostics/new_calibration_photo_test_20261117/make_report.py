from pathlib import Path
import csv,json,hashlib,zipfile
import cv2,numpy as np
out=Path(__file__).resolve().parent
summary=json.loads((out/'summary.json').read_text());inputs=json.loads((out/'input_manifest.json').read_text())
results=[json.loads((out/('result_'+p+'.json')).read_text()) for p in summary['included_photos']]
assert summary['excluded_photos']==['0003','0004'] and len(results)==3
panels=[]
for r,source in zip(results,inputs):
 im=cv2.rotate(cv2.imread(source['original_absolute_path']),cv2.ROTATE_180)
 x0,y0,x1,y1=400,140,740,500
 crop=im[y0:y1,x0:x1].copy()
 eps=r['pixels']['rail_endpoints_px']
 if eps:
  points=[tuple(np.rint(np.array(p)-[x0,y0]).astype(int)) for p in eps];cv2.line(crop,*points,(0,255,0),1,cv2.LINE_AA)
 for color,bgr in [('blue',(255,140,0)),('yellow',(0,210,255))]:
  point=tuple(np.rint(np.array(r['sliders'][color]['pixel_xy_upright'])-[x0,y0]).astype(int));cv2.drawMarker(crop,point,bgr,cv2.MARKER_CROSS,12,1)
 crop=cv2.resize(crop,(680,720),interpolation=cv2.INTER_LINEAR);header=np.zeros((145,680,3),np.uint8)
 lines=[f'Photo {r["photo"]} | new video31 K/D + pose 0',
  f'Blue   ref {r["sliders"]["blue"]["reference_world_x_mm"]:.2f}, X {r["sliders"]["blue"]["detected_world_x_mm"]:.3f}, err {r["sliders"]["blue"]["error_world_x_mm"]:+.3f}',
  f'Orange ref {r["sliders"]["yellow"]["reference_world_x_mm"]:.2f}, X {r["sliders"]["yellow"]["detected_world_x_mm"]:.3f}, err {r["sliders"]["yellow"]["error_world_x_mm"]:+.3f}']
 for i,line in enumerate(lines):cv2.putText(header,line,(12,30+43*i),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
 panels.append(np.vstack([header,crop]))
blank=np.zeros_like(panels[0]);lines=['WORLD X / mm | errors = detection - reference',
 'C=(230.1144, 98.5561, -98.1882) mm',
 'Raw calibration -> upright K/D/R together',
 'Raw image rotated 180 degrees exactly once',
 'Same source images and detected shaft pixels',
 'No fitting or added position compensation',
 'Original photo 3 and 4 remain excluded',
 'MAE 4.637 -> 2.384 mm; max 5.200 -> 2.985',
 'All 6 errors still negative; 0/6 within 0.5 mm',
 'Lens/focus/camera unchanged: user confirmed']
for i,line in enumerate(lines):cv2.putText(blank,line,(13,48+i*55),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
cv2.imwrite(str(out/'新标定导轨检测对照.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack([panels[2],blank])]),[cv2.IMWRITE_JPEG_QUALITY,95])
rows=[]
for r in results:
 for color in ['blue','yellow']:
  a=r['sliders'][color]
  rows.append('|%s|%s|%.2f|%.3f|%+.3f|%.3f|%+.3f|' % (str(int(r['photo'])),a['color'],a['reference_world_x_mm'],a['previous_world_x_mm'],a['previous_error_world_x_mm'],a['detected_world_x_mm'],a['error_world_x_mm']))
relative={}
for color,a,b in [('blue',results[0],results[2]),('yellow',results[0],results[1])]:
 p=a['sliders'][color];q=b['sliders'][color]
 relative[color]=dict(reference_delta_mm=q['reference_world_x_mm']-p['reference_world_x_mm'],detected_delta_mm=q['detected_world_x_mm']-p['detected_world_x_mm'])
summary['relative_motion_examples']=relative
summary['mean_absolute_error_reduction_fraction']=1-summary['metrics']['combined']['new']['mae_mm']/summary['metrics']['combined']['previous']['mae_mm']
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
report='''# 新内外参在最近地面实拍图上的离线重检

**新标定明显改善了位置误差，但本组尚未达到±0.5mm。** 同一组6个已知杆位，平均绝对误差4.637→2.384mm，RMSE4.662→2.438mm，最大绝对偏差5.200→2.985mm；6个读数仍全部偏小。未用这些参考点拟合参数，也未加统一偏移补偿。

## 输入与公平比较

采用用户上传camera_world_sensor_raw.json（SHA 6f1673f602ffe0cdc0b98c20ceecd7cd79f292c0ae69128073dc10c0ed3a125e），逐字段确认即上次审查的候选0，其K/D与新calibration (2).json一致，规范化JSON源哈希一致。光心C=(230.1143765,98.5560549,−98.1882375)mm，实际video31，原图1920×1080、sensor_raw、左手世界。

图片为最近用户提供session_20260510_222231_76459ec8.zip中的第1、2、5张，3/4保持用户排除，不再读取它们的图像/检测。三张原PNG SHA分别与各自元数据和前次检测记录一致；每图备注世界X作为原有手工参考，量测不确定度未提供。原导出JSON中的旧reference_geometry不参与本次换算。用户本轮明确确认：**镜头、焦点和相机位置都未改变**，所以这组旧图可作为新标定的回放验证数据。

基线使用上轮“新测名义外参+借用video22旧内参”的同一份used_calibration；没有混用早期不同X轴/不同止挡参考。对同一图重新运行现有检测器，杆轴/端点全部像素字段与上次完全相同。原图自动转正与手动转正already_rotated=true也逐项一致。算法源码三项SHA在重检前后未变。

## 原图与转正外参配套

当前检测流程先将倒装原图转正180°。输入文件属于sensor_raw，不能直接当转正外参。此次在离线副本同时变换：cx'=1919−cx，cy'=1079−cy，p1/p2反号，R'=R_raw×diag(−1,−1,1)，光心不变。实际使用K为fx704.2315208、fy700.8145421、cx950.9108070、cy463.0003077。显式完整标定绕过历史图像斜率姿态补偿，没有额外拟合倾角或偏移。

另一路独立核验完全使用上传的原图K/D/R：把检测像素反转回sensor_raw，以独立Newton法反解五参数Brown畸变，再以世界射线与Y=181.5平面求交。两条路径6点的最大XYZ差仅4.09e−11mm，确认不是180°方向或切向畸变符号混用。此数值是数学一致性检查，不是实测精度。

## 逐点世界X结果

单位mm，偏差=识别−备注参考；“前次”与“本次”使用相同图片、相同像素、相同世界参考。

|照片|颜色|参考X|前次识别X|前次偏差|本次识别X|本次偏差|
|---|---|---:|---:|---:|---:|---:|
TABLE

|统计（mm）|前次|本次|
|---|---:|---:|
|蓝杆平均偏差|−5.018|−2.801|
|橙杆平均偏差|−4.257|−1.968|
|整体平均绝对误差|4.637|2.384|
|整体RMSE|4.662|2.438|
|最大绝对偏差|5.200|2.985|
|±0.5mm内数量|0/6|0/6|

平均绝对误差减少约48.6%。3/3张双杆像素可靠，按相同固定行程和原2.5mm超止挡容差也3/3张任务视觉可靠；这些筛选不代表绝对位置精度已通过。这里统计的是独立手工世界位置参考误差，不是机械臂运动验收结果。

## 现象与边界

误差由全部偏小4～5mm改善为偏小1.72～2.99mm；同向残差仍存在。蓝第1→5图参考位移−8.50mm，新识别−8.50364mm；橙第1→2图参考+20.00mm，新识别+20.03004mm。因此本组相对位移接近参考，而绝对位置仍未对齐。仅验证两杆间距或位移不能认证绝对坐标。

由于像素完全未变，这一轮改善来源于新K/D/C/R的合并作用；不能只归因光心X，也不能用残差直接推定还有多少入瞳深度。剩余偏差可能涉及已知棋盘世界位置/姿态基准、导轨轴心Y平面、杆轴像素/实测参考点定义或标定覆盖；仅此三张不能唯一分解。未对任何一项反调参数凑合结果。

当前世界平面仍沿用轴心Y181.5、挡边Y172.5；后者来自旧9mm关系，未独立复测。肩轴S=(65,26.5,−64.5)、世界→局部符号[−1,1,1]不变，X_arm=65−X_world。每图JSON还保存导轨端点XYZ和局部坐标，但本组没有端点真值，故不以它们宣称槽长精度通过。

后续宜固定棋盘核对真实O/U/V世界坐标与姿态，并在与杆轴相同的区域/深度用未参与求解的已知点复核；内参上方/边缘覆盖不足的提示仍有效。此次按用户提供的原始新内外参原样测试，没有改阈值、加补偿或替换标定。

## 标注与可复核输出

![新标定导轨检测对照](IMAGE)

绿色为图像检测的导轨端点/参考线，蓝/黄十字为检测杆轴像素。原PNG未改，仅输出测量标注图。detected_0001/0002/0005.jpg为完整视场标注，result_*.json保存参考、新旧世界坐标、局部坐标、全部像素、独立原图射线核验和可靠性。results.csv、summary.json提供统计。

provided_camera_world_sensor_raw.json/provided_intrinsics.json是本次用户输入原样备份；used_calibration_scene_upright.json是仅作配套180°变换的实际离线参数；used_vision_config.json、used_source/和conversion_receipt.json记录处理配置及来源。输入原图绝对路径/哈希在input_manifest.json；ZIP不重复包含原PNG（用户原包和工作区原图保留）。

全部离线：未打开实时相机/串口、未运动、未改PID/offset、生产源码或运行配置，未向板端部署或激活新标定。只有本目录的报告、离线参数副本和测量结果新增。
'''.replace('TABLE','\n'.join(rows)).replace('IMAGE',str(out/'新标定导轨检测对照.jpg'))
(out/'新标定地面实拍测试报告.md').write_text(report)
names=['新标定地面实拍测试报告.md','新标定导轨检测对照.jpg','results.csv','summary.json',
 'provided_camera_world_sensor_raw.json','provided_intrinsics.json','used_calibration_scene_upright.json',
 'used_vision_config.json','conversion_receipt.json','exclusions.json','input_manifest.json','replay.py','make_report.py']
names += ['result_'+p+'.json' for p in summary['included_photos']]
names += ['detected_'+p+'.jpg' for p in summary['included_photos']]
names += [str(p.relative_to(out)) for p in (out/'used_source').rglob('*') if p.is_file()]
manifest={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
zip_path=out/'新标定地面实拍测试结果.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for n in names+['manifest.json']:z.write(out/n,n)
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 assert not any('0003' in n or '0004' in n for n in z.namelist())
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
(out/'package_receipt.json').write_text(json.dumps(dict(path=str(zip_path),bytes=zip_path.stat().st_size,sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),crc_pass=True,all_members_sha256_verified=True,original_PNGs_included=False),ensure_ascii=False,indent=2))
print(out/'新标定地面实拍测试报告.md');print(summary['mean_absolute_error_reduction_fraction']);print(relative)

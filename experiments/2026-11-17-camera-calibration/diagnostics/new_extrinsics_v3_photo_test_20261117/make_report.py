from pathlib import Path
import hashlib,json,zipfile,sys
import cv2,numpy as np
cv2.setNumThreads(1)
out=Path(__file__).resolve().parent
summary=json.loads((out/'summary.json').read_text())
verification=json.loads((out/'extrinsics_verification.json').read_text())
sources=json.loads((out/'input_manifest.json').read_text())
results=[json.loads((out/f'result_{p}.json').read_text()) for p in summary['included_photos']]
assert summary['included_photos']==['0001','0002','0005'] and summary['excluded_photos']==['0003','0004']
relative={}
for color,ia,ib in [('blue',0,2),('yellow',0,1)]:
 a,b=results[ia]['sliders'][color],results[ib]['sliders'][color]
 relative[color]=dict(from_photo=results[ia]['photo'],to_photo=results[ib]['photo'],
  reference_delta_mm=b['reference_world_x_mm']-a['reference_world_x_mm'],
  detected_delta_mm=b['detected_world_x_mm']-a['detected_world_x_mm'])
summary['relative_motion_examples']=relative
summary['mean_absolute_error_reduction_fraction']=1-summary['metrics']['combined']['new']['mae_mm']/summary['metrics']['combined']['previous']['mae_mm']
summary['all_new_errors_positive']=all(i['error_world_x_mm']>0 for r in results for i in r['sliders'].values())
summary['rail_lip_y_changed_from_mm']=verification['previous_rail_lip_y_mm']
summary['rail_lip_y_changed_to_mm']=verification['rail_lip_plane_y_mm']
# Confirm the explicitly supplied lip plane only changes endpoint conversion,
# while slider centers still use the unchanged shaft plane Y=181.5mm.
workspace=out.parents[1]
sys.path.insert(0,str(workspace/'arm_real_control'))
from vision_tools.pixel_to_world import convert_detections
cal=json.loads((out/'used_calibration_scene_upright.json').read_text())
diagnostic=dict(cal,rail_lip_y_mm=verification['previous_rail_lip_y_mm'])
lip_effect=[]
for r in results:
 alternative=convert_detections(r['pixels'],diagnostic)
 for color in ['blue','yellow']:
  a=np.array(alternative['sliders'][color]['world_xyz_mm'])
  b=np.array(r['world']['sliders'][color]['world_xyz_mm'])
  lip_effect.append(float(np.max(np.abs(a-b))))
assert max(lip_effect)==0
summary['lip_plane_change_effect_on_slider_XYZ_mm']=max(lip_effect)
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

# Three approved raw photographs only, cropped and labelled as diagnostic views.
panels=[]
for r,source in zip(results,sources):
 im=cv2.rotate(cv2.imread(source['original_absolute_path']),cv2.ROTATE_180)
 x0,y0,x1,y1=400,140,740,500
 crop=im[y0:y1,x0:x1].copy()
 eps=r['pixels']['rail_endpoints_px']
 if eps:
  pts=[tuple(np.rint(np.array(p)-[x0,y0]).astype(int)) for p in eps]
  cv2.line(crop,*pts,(0,255,0),1,cv2.LINE_AA)
 for color,bgr in [('blue',(255,140,0)),('yellow',(0,210,255))]:
  pt=tuple(np.rint(np.array(r['sliders'][color]['pixel_xy_upright'])-[x0,y0]).astype(int))
  cv2.drawMarker(crop,pt,bgr,cv2.MARKER_CROSS,12,1)
 crop=cv2.resize(crop,(680,720),interpolation=cv2.INTER_LINEAR)
 header=np.zeros((145,680,3),np.uint8)
 lines=[f'Photo {int(r["photo"])} | 34-image intrinsics + new pose 0']
 for color,label in [('blue','Blue'),('yellow','Orange')]:
  a=r['sliders'][color]
  lines.append(f'{label}: ref {a["reference_world_x_mm"]:.2f}, X {a["detected_world_x_mm"]:.3f}, error {a["error_world_x_mm"]:+.3f}')
 for i,line in enumerate(lines):cv2.putText(header,line,(12,30+43*i),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
 panels.append(np.vstack([header,crop]))
blank=np.zeros_like(panels[0])
lines=['World X / mm; error = detection - reference',
 'Candidate 0: reprojection RMS 0.208 px',
 'Candidate 1: reprojection RMS 1.792 px',
 'C = (236.630, 98.327, -101.507) mm',
 'Same images, detected pixels and reference X',
 'Raw image rotated 180 degrees exactly once',
 'K / D / R converted together; C unchanged',
 'Previous MAE 2.384 -> new MAE 1.980 mm',
 'Max abs error 2.985 -> 2.595 mm',
 'All six X errors positive; 0/6 within 0.5 mm',
 'Photo 3 and 4 remain excluded',
 'No fitting, compensation or deployment']
for i,line in enumerate(lines):cv2.putText(blank,line,(13,45+i*51),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
img=out/'新外参实拍检测对照.jpg'
assert cv2.imwrite(str(img),np.vstack([np.hstack(panels[:2]),np.hstack([panels[2],blank])]),[cv2.IMWRITE_JPEG_QUALITY,95])

rows=[]
for r in results:
 for color in ['blue','yellow']:
  a=r['sliders'][color]
  rows.append('|%s|%s|%.2f|%.3f|%+.3f|%.3f|%+.3f|' % (int(r['photo']),a['color'],a['reference_world_x_mm'],a['previous_world_x_mm'],a['previous_error_world_x_mm'],a['detected_world_x_mm'],a['error_world_x_mm']))
report='''# 新外参候选检查与实拍验证

**应选候选0，用户导出的camera_world_sensor_raw.json已经对应候选0。** 新内外参在最近三张实拍图上平均绝对误差由2.384降至1.980mm，最大误差2.595mm；仍未达到±0.5mm，且误差从上一版全部偏小转为本版全部偏大。因此选对候选不等于世界坐标精度已经合格。

## 候选与内参配套检查

|指标|候选0|候选1|
|---|---:|---:|
|光心C世界XYZ/mm|(236.630,98.327,−101.507)|(236.360,208.750,20.097)|
|重投影RMS/px|0.2083|1.7924|
|最大单角点误差/px|0.3960|4.8668|
|光轴相对世界−X偏角|2.271°|41.294°|

候选1误差是候选0约8.61倍，光心Y/Z和光轴也与已有安装信息明显不符；不属于两个同样合理的近似解。两者均正深度且R正交，所以不能仅靠“棋盘在前方”或正交性选择。

本版使用video31/m01_f_imx415 4-001a/rkisp1-vir0、1920×1080 sensor_raw。K/D、RMS、34张来源信息与上一轮calibration (3).json一致：fx716.218990、fy709.895404、cx962.403066、cy549.185229。原图/180°转正两套K/D/R也全部配套，世界为用户确认的左手坐标（仅X反向），正交矩阵行列式−1符合坐标基转换，不能按右手旋转矩阵判成错误。

初次严格比较内参源JSON哈希时发现不同：上传文件规范化SHA为1c35c6b…，外参引用0912d6ae…。已查明并精确复现：网页解析再序列化JSON会把6.0、0.0等整值浮点写为6、0，数值不变而Python规范化文本哈希不同。只做此整数表示转换后精确得到0912d6ae…；全部K/D数值、设备与模式一致。这不是使用了另一套参数。证明见intrinsic_source_normalization_check.json；未改用户输入。

只读板端对应token ccb106f7aee4b3bd保存会话，上传extrinsics与板端原文件逐字节一致。原图SHA4962ba0efa5da5581d386f6e0a00f875b0b3ff0c11891128ca340e46128f2ac6与capture记录一致，读取前后元数据不变。从保存原始角点独立调用IPPE，复现两候选C/R；另直接以世界角点、C/R、K/D重投影复现每角点误差，无tvec/光心混淆。

## 已知世界棋盘的输入约束

本次O=(9.2,174.4,−25.8)，U=(9.2,132.4,−25.8)，V=(9.2,174.4,−55.8)mm；OU=42、OV=30mm且正交。点选编号47/40/7与倒装原图双向反转的排列相符。输入要求整块板X=9.2、OU严格沿−Y、OV严格沿−Z，实际必须满足这些姿态约束。

原图仍可见手托棋盘；图像不能独立证明实际三点位置/方向与输入完全一致，尤其不能用较低重投影误差认证世界轴对齐。若实际板面倾斜而仍输入严格轴对齐坐标，误差会进入外参。优先用刚性机械基准固定贴平后复核三点，并以另一已知位置检验同一个相机C/R。不要仅凭新光心X较机械尺量值大就判错或把C强改回镜头前端；机械镜头点不是独立光学中心真值。

## 最近实拍照片回放

继续使用用户最近提供76459ec8原始PNG第1、2、5张，第3、4张仍按用户要求排除，本轮未读它们图像或运行检测。以此前用户明确确认“镜头、焦点和相机位置都未改变”为旧图回放前提；本轮没有独立复测物理安装位置。如果此后改变过镜头/焦点/安装，必须在最终安装姿态补拍验证。

各图PNG哈希与原元数据、上轮记录一致。再次运行相同检测器，所有像素字段与前次完全一致；原图自动180与手动180+already_rotated也一致。导出JSON旧reference_geometry忽略，参考X仍只取用户照片备注，量测不确定度未提供。没有用参考点拟合任何内外参、比例或统一偏移。

新sensor_raw参数在离线副本同时变换为转正参数：cx'=1919−cx、cy'=1079−cy、p1/p2反号、R'=Rraw×diag(−1,−1,1)，C不变。另用完全独立Newton反解Brown畸变，在原图坐标求世界射线与Y=181.5平面相交，与正式换算最大XYZ差约1.40e−10mm。这只是数值一致性检查，不是物理精度。完整三项检测/换算源码SHA前后未变。

|照片|杆|参考世界X|上一版X|上一版偏差|本次X|本次偏差|
|---|---|---:|---:|---:|---:|---:|
TABLE

单位mm，偏差=识别−参考。“上一版”指上一份本机30张内参加候选0外参，而不是早期借用video22内参的版本。

|统计|上一版|本次|
|---|---:|---:|
|蓝杆平均偏差/mm|−2.801|+1.690|
|橙杆平均偏差/mm|−1.968|+2.271|
|整体平均绝对误差/mm|2.384|1.980|
|整体RMSE/mm|2.438|2.023|
|最大绝对偏差/mm|2.985|2.595|
|±0.5mm内|0/6|0/6|

总体平均绝对误差降低约16.9%；蓝杆改善，橙杆平均绝对误差反而由1.968升至2.271mm。不能把总指标小幅下降解释成全行程精度已充分改善。6点相对上一版全部增加约4.13～4.50mm，从同向偏负跨到同向偏正，提示绝对世界基准/几何仍值得优先复核；仅6点不足以唯一判断原因。

相对位移：蓝1→5参考−8.50mm、本次−8.53689mm；橙1→2参考+20.00mm、本次+20.21222mm。相对位移较接近不代表绝对坐标无偏。3/3图双杆像素及现有任务筛选均可靠，其physical-stop筛选容差是2.5mm，不等同于±0.5mm精度认证。

## 参数变化与后续重点

新光心相较上一版变化(+6.515,−0.229,−3.318)mm，K/D和R也同时变化；本次改善/偏差反向来自整套映射，不能只归因一个坐标或认为确有相机移动。新文件轴心Y仍181.5，挡边Y由172.5改为173.5，按用户新输入原样使用。离线单独切换挡边Y确认杆轴坐标完全不变，它只改变导轨端点换算，不能用此解释/补偿本次杆轴X偏差；需实量确认新的挡边高度。

当前优先项：固定并核实O/U/V世界位姿与打印尺度；核实杆轴Y=181.5及参考X均对应轴心、同一世界基准；用未参与求解的已知点在杆工作区域验证，再判断是全局标定偏差还是杆轴测量定义问题。不要直接把平均+1.98mm写成补偿，也不靠调PID/电机零点修正静态照片误差。

本轮仅保存文件下载与离线验证，未开启实时相机/串口、未运动、未改生产代码/配置/PID/offset、未部署或激活外参。

## 图与证据

![新外参实拍检测对照](IMAGE)

绿色为检测导轨线/端点，蓝/黄十字为杆轴检测像素。原PNG未改。完整视场标注detected_0001/0002/0005.jpg；逐图result_*.json包含新旧X、参考值、局部坐标、原图射线核验及可靠性；results.csv和summary.json给出统计。provided_*是原样输入，used_*是离线实际配套参数/源码，extrinsics_verification.json保存独立求解核验，fetch_receipt.json保存板端证据来源。

方法参考：[OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)。独立世界精度结论以本组参考坐标比较为准，不能由PnP训练图像残差直接替代。
'''.replace('TABLE','\n'.join(rows)).replace('IMAGE',str(img))
(out/'新外参检查与实拍验证.md').write_text(report)

# Package explicit non-secret artifacts. Read-only SSH helper scripts are excluded.
names=['新外参检查与实拍验证.md','新外参实拍检测对照.jpg','results.csv','summary.json',
 'provided_extrinsics.json','provided_camera_world_sensor_raw.json','provided_intrinsics.json','provided_input_manifest.json',
 'used_calibration_scene_upright.json','used_vision_config.json','conversion_receipt.json','exclusions.json',
 'input_manifest.json','extrinsics_verification.json','fetch_receipt.json','intrinsic_source_audit.json',
 'intrinsic_source_normalization_check.json','replay.py','verify_extrinsics.py','make_report.py']
names += [f'result_{p}.json' for p in summary['included_photos']]
names += [f'detected_{p}.jpg' for p in summary['included_photos']]
names += [str(p.relative_to(out)) for p in (out/'used_source').rglob('*') if p.is_file()]
names += ['board_evidence/'+p for p in ['capture.json','extrinsics.json','sensor_raw.png','corners.jpg']]
manifest={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
zpath=out/'新外参检查与实拍验证结果.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for n in names+['manifest.json']:z.write(out/n,n)
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 assert not any('0003' in n or '0004' in n or 'fetch_capture' in n or 'audit_intrinsic_source.py' in n for n in z.namelist())
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
(out/'package_receipt.json').write_text(json.dumps(dict(path=str(zpath),bytes=zpath.stat().st_size,sha256=hashlib.sha256(zpath.read_bytes()).hexdigest(),crc_pass=True,member_hashes_verified=True,SSH_helpers_excluded=True),ensure_ascii=False,indent=2))
print(json.dumps(dict(report=str(out/'新外参检查与实拍验证.md'),image=str(img),zip=str(zpath),relative=relative,MAE_reduction_fraction=summary['mean_absolute_error_reduction_fraction']),ensure_ascii=False,indent=2))

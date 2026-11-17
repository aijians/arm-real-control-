from pathlib import Path
import hashlib,json,zipfile
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
out=Path(__file__).resolve().parent;prev=out.parent/'new_extrinsics_v3_photo_test_20261117'
s=json.loads((out/'solution_report.json').read_text());c=json.loads((out/'corner_residuals.json').read_text())
ext=json.loads((prev/'provided_extrinsics.json').read_text())
im=cv2.cvtColor(cv2.imread(str(prev/'board_evidence/sensor_raw.png')),cv2.COLOR_BGR2RGB)
obs=np.array([r['observed_px'] for r in c]);strict=np.array([r['strict_projected_px'] for r in c])
original=np.array(ext['candidates'][0]['projected_pixels'])
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
lo=np.floor(obs.min(axis=0)-25).astype(int);hi=np.ceil(obs.max(axis=0)+25).astype(int)
fig,axs=plt.subplots(1,2,figsize=(12,5.7),dpi=150)
for ax,pr,title in zip(axs,[original,strict],['原自由外参：RMS 0.208 px','固定 Y/Z 与光轴 −X：RMS 5.075 px']):
 ax.imshow(im[lo[1]:hi[1],lo[0]:hi[0]],extent=(lo[0]-.5,hi[0]-.5,hi[1]-.5,lo[1]-.5))
 ax.scatter(obs[:,0],obs[:,1],s=35,marker='+',c='#00C7D9',linewidths=1.3,label='观测角点')
 ax.scatter(pr[:,0],pr[:,1],s=30,marker='x',c='#E22D2D',linewidths=1.,label='计算投影')
 for a,b in zip(obs,pr):ax.plot([a[0],b[0]],[a[1],b[1]],c='#E22D2D',lw=.8)
 for index,label in [(0,'O'),(7,'U'),(40,'V')]:
  ax.annotate(label,obs[index],xytext=(6,6),textcoords='offset points',fontsize=12,color='#164BFF',weight='bold')
 ax.set_title(title,fontproperties=font,fontsize=13);ax.set_xlabel('sensor_raw u / px');ax.set_ylabel('sensor_raw v / px')
 ax.legend(prop=font,loc='upper left',fontsize=9);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
fig.suptitle('相同内参、同一张棋盘、相同已知世界位置',fontproperties=font,fontsize=15)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'棋盘约束重投影对照.png');plt.close(fig)
rows=[]
for r in s['photo_results']:
 rows.append('|%d|%s|%.2f|%.3f|%+.3f|' % (int(r['photo']),r['color'],r['reference_X_mm'],r['world_X_mm'],r['error_X_mm']))
report='''# 按用户确认的安装与世界坐标约束重新求解外参

本轮将用户确认的机械位置及棋盘世界坐标作为固定输入，不再通过放开相机Y/Z或光轴来吸收误差。已完成约束求解和实拍验证，但在现有内参保持不变的条件下，**得到的外参未通过验证，不能作为有效的新标定激活。**

## 固定输入与自由参数

- 相机光心Cy=97mm、Cz=−102.8mm；光轴严格为世界[−1,0,0]。
- O=(9.2,174.4,−25.8)、U=(9.2,132.4,−25.8)、V=(9.2,174.4,−55.8)mm；按用户确认接受，OU42/OV30mm、左手世界。
- 采用同一份34张内参K/D，fx716.218990、fy709.895404、cx962.403066、cy549.185229；本轮正式外参求解没有修改内参。
- 只剩两个外参自由度：光心Cx、绕光轴的安装转角。垂直安装约束光轴，不会单独确定绕轴转角。
- 机械臂基点采用最新S=(65,26.5,−64.8)，只用于世界→局部坐标，不参与相机位姿拟合。

## 重新求解结果

```text
C_world_mm = [222.423014, 97.000000, -102.800000]
光轴        = [-1, 0, 0]
R_world_from_camera（sensor_raw） =
[[ 0,          0,         -1],
 [ 0.99645862, 0.08408459,  0],
 [-0.08408459, 0.99645862,  0]]
```

该参数化严格保持正交、左手世界转换行列式−1和光轴−X；不是只改R的第三列。绕轴转角−4.8234°是相对于本报告明确矩阵基准的数值表达，不是对用户坐标方向的新修改。原图与180°转正版本同时转换K/D/R，C不变。

用归一化像平面线性最小二乘独立初始化，再以13组不同Cx/绕轴初值做像素残差最小二乘，13组都到达同一数值解。重新求解结果与上一轮仅作诊断的严格约束方案F一致，没有单一初值导致的新解漂移；这不是新的低误差解。

|验证项|结果|
|---|---:|
|同一棋盘重投影RMS|5.0747 px|
|最大角点重投影误差|8.5169 px|
|最近实拍6点X平均绝对误差|9.8201 mm|
|最近实拍6点最大绝对误差|10.4892 mm|
|±0.5mm内点数|0/6|

最近实拍仍仅第1、2、5张，第3、4张保持排除。重新读取原PNG核对SHA并再次运行检测，像素与前次完全相同。滑杆参考只用于验证，没有用于拟合外参。单位mm，偏差=识别−原照片备注参考：

|照片|杆|参考世界X|约束外参识别X|偏差|
|---|---|---:|---:|---:|
TABLE

## 额外复核与定位

从棋盘原图ROI以全分辨率SB重新提取角点，和保存角点差异RMS仅0.1501px、最大0.5041px；用重新提取的角点再做严格约束求解，棋盘RMS仍5.0913px。因此这5px量级冲突不能由本轮两种角点提取的差异解释。

在接受用户几何条件的前提下，当前固定内参与这些图像的对应应优先复核。本轮另做灵敏度诊断：保持焦距与D不变，仅在严格几何条件下临时允许主点变化，能把该棋盘RMS由5.075降至0.446px，对应假设cx≈955.442、cy≈573.293，较原值约−6.96/+24.11px。它只表明主点/像素坐标关系能吸收较大部分冲突，不足以证明主点就应改成这两个数；单张世界棋盘不能替代多姿态内参标定，未导出或应用这份假设内参，也未用滑杆参考去调它。

已有文件层面的相机标识、1920×1080、sensor_raw与180°配套约定一致。当前本地标定程序的正式图像路径无应用层裁切/翻转/缩放，预览/初检缩小后角点回到全分辨率，外参冻结图与角点为同帧；这不等于已经独立认证所有底层相机模式。后续应使用相同最终模式、多姿态棋盘数据复核K/D对可信几何的解释能力，不再把自由外参的低RMS当作机械条件的修正依据。

此处无法仅凭一张图唯一诊断内参哪一项错误；也不能为了通过而擅改用户确认的机械位置/棋盘世界点。正式输出保持所有约束，验证失败如实记录。

## 输出与应用状态

![棋盘约束重投影对照](IMAGE)

camera_world_sensor_raw_constrained_failed.json和camera_world_scene_upright_constrained_failed.json保存这次严格约束的数学解，文件名与validation_status明确标记验证失败，仅用于复核。offline_vision_config_failed.json是使用该解和新基点的离线副本。它们没有替换用户原标定或任何运行配置。

solution_report.json、corner_residuals.json、photo_results.csv和solve.py记录全部数值、13初值结果、原图哈希、独立角点提取及主点诊断。棋盘原图沿用上次已验SHA的原始证据，不修改。未访问板端、开实时相机/UART、运动、写PID/offset、修改生产源码、部署或GitHub发布。
'''.replace('TABLE','\n'.join(rows)).replace('IMAGE',str(out/'棋盘约束重投影对照.png'))
(out/'严格安装约束外参重求结果.md').write_text(report)
names=['严格安装约束外参重求结果.md','棋盘约束重投影对照.png','camera_world_sensor_raw_constrained_failed.json',
 'camera_world_scene_upright_constrained_failed.json','offline_vision_config_failed.json','solution_report.json','corner_residuals.json','photo_results.csv','solve.py','make_report.py']
manifest={p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in names}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
zpath=out/'严格约束外参求解_未通过验证.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for n in names+['manifest.json']:z.write(out/n,n)
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
(out/'package_receipt.json').write_text(json.dumps(dict(bytes=zpath.stat().st_size,sha256=hashlib.sha256(zpath.read_bytes()).hexdigest(),crc_pass=True,member_hashes_verified=True,validation_passed=False),ensure_ascii=False,indent=2)+'\n')
print(out/'严格安装约束外参重求结果.md')

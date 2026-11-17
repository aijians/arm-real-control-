from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
out=Path(__file__).parent
s=json.loads((out/'summary_initial.json').read_text());r=json.loads((out/'recalculation.json').read_text());cal=json.loads((out/'input_intrinsics.json').read_text());ext=json.loads((out/'input_extrinsics.json').read_text());samples=json.loads((out/'board_evidence/samples.json').read_text())
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.unicode_minus':False})
# Use explicit font properties for Chinese text, independent of installed font caches.
fig,ax=plt.subplots(1,2,figsize=(12,4.6),dpi=160)
for m in samples['samples']:
 p=np.array(m['corners']);ax[0].scatter(p[:,0],p[:,1],s=2,c='#24749e',alpha=.2)
centers=np.array([m['center_px'] for m in r['per_image']]);ax[0].scatter(centers[:,0],centers[:,1],s=18,c='#16496b',label='30 sample centers')
ax[0].scatter(*centers[17],s=60,c='#d95747',label='Sample 18')
for x in (640,1280):ax[0].axvline(x,c='#cccccc',lw=.8)
for y in (360,720):ax[0].axhline(y,c='#cccccc',lw=.8)
ax[0].set(xlim=(0,1920),ylim=(1080,0),xlabel='Sensor raw u / px',ylabel='Sensor raw v / px',title='Intrinsic sample coverage (30 images)')
ax[0].set_aspect('equal');ax[0].legend(loc='upper right',fontsize=8)
values=np.array([m['rms_px'] for m in r['per_image']]);colors=['#3c83a7']*30;colors[17]='#d95747'
ax[1].bar(np.arange(1,31),values,color=colors,width=.8)
ax[1].axhline(cal['rms_px'],color='#d79430',ls='--',label='Overall RMS 0.344 px')
ax[1].annotate('18: 1.268 px',xy=(18,values[17]),xytext=(20,1.3),arrowprops={'arrowstyle':'-','color':'#444444'})
ax[1].set(xlabel='Sample number',ylabel='Per-image reprojection RMS / px',title='Reprojection errors',ylim=(0,1.5))
ax[1].set_xticks([1,5,10,15,18,20,25,30]);ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(out/'采样覆盖与误差.png');plt.close(fig)
report='''# 本次内参与世界外参检查

建议选择**候选0（列表中的第一个）**。本次两个解的差异足够明显，候选1可排除。候选0具备作为待验证外参的依据，但单张已知棋盘的像素误差不能独立保证世界毫米精度。

本轮只读用户文件和开发板历史采样，未开新相机流、未改代码/标定参数或机械臂配置，未激活任何候选。

## 数据与过程核对

用户提供的calibration (2).json对应video31 / m01_f_imx415 4-001a / rkisp1-vir0，1920×1080，sensor_raw，未预先去畸变。实际30张8×6内角点、6mm方格，已是当前实体相机的新标定，不再是原video22借用值。外参报告内嵌K/D与本次内参完全一致，source_sha256按程序的规范化JSON计算也一致；该哈希本来就不同于文件原始字节SHA，不是文件损坏。

新SSH只读定位本组目录 `/var/lib/camera-calibration-video22/20260510T170657Z_b0a8ad`。根目录名保留旧video22服务设置，不代表实际相机错误：samples.json、calibration.json、capture.json均声明video31及同一实体。板端内参与用户内参字节完全一致；本次外参原图token=f471d02363658eb2，PNG SHA匹配capture.json。板端同token的外参报告与用户上传报告只有created_utc重算时间不同，其他内容全部一致；本报告以用户上传版为准。板端系统日期不作为实际实验日期。

用保存的30×48角点独立重算calibrateCamera，RMS复现0.3440429924px、K/D复现到浮点精度；用capture角点/三点输入重算IPPE复现两候选，再独立用世界C/R把48个世界角点投影回图像，重投影误差亦复现。未发现错用相机内参、180重复变换、左手world矩阵镜像或tvec误当C的问题。

## 两个候选的选择

|项目|候选0|候选1|
|---|---:|---:|
|重投影RMS / px|0.172693|3.324609|
|最大角点误差 / px|0.353116|9.811820|
|光心X / mm|230.114|229.647|
|光心Y / mm|98.556|207.994|
|光心Z / mm|−98.188|34.567|
|光轴偏离世界−X|2.342°|49.435°|

候选1 RMS是候选0的19.25倍，Y/Z与此前尺量镜头位置(228.99,97,−104.22)相差约111/139mm，光轴也偏离已知朝向约49°。候选0方向约为(−0.999165,+0.008608,+0.039944)，接近实际朝世界−X的安装约束。这两项证据相互支持选0，而不是仅看哪个X接近旧值。两候选所有棋盘角点都在相机前方，单独的正深度检查无法替代上述判别。

planar_ambiguity=false与数据一致：两个候选RMS差3.152px，远大于程序提示近似双解的.15px阈值。IPPE可以返回多个平面解，但返回两个并不代表两者质量相同。参考[OpenCV PnP说明](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)。

候选0相对原尺量镜头点的差为(+1.124,+1.556,+6.032)mm。这里只把尺量值作为粗安装约束，不当独立精确光心真值。Z差约6mm，不能仅靠“入瞳在镜头后方”这一项解释，需复核三点世界坐标、棋盘姿态/平整度及镜头/坐标测量基准。

## 内参采集质量

30张总体RMS0.344px，单图中位数0.245px；29张RMS低于1px。第18张RMS1.268px，单点最大4.684px，占整组角点残差平方和约45.3%；第22张0.505px次之。查看原图，第18张下部黑白边界存在反光、局部细节不佳；原分辨率SB独立检查也未成功检出完整棋盘（第22张同样未成功，不能单以此作判废）。建议重拍高误差图，避免亮部反光，不仅依赖全局清晰度指标。

诊断对照中仅暂时排除第18张重新求解（不改原文件/正式结果）：RMS降至0.258731px，fx/fy分别改变+.352/+.340px，主点改变(−.214,−.561)px；同一外参图得到候选0光心变化(+.1129,+.0023,+.0324)mm，合计0.1175mm。因此第18张值得改善，但它不是候选0/1取舍的原因，也不足以单独解释6mm量测差。排除后训练RMS变低不能当独立精度验证。

更需补充的是姿态和位置覆盖：棋盘中心仅覆盖九宫格4/6/7/8（中心和下排），上排及中排两侧欠缺。所有角点范围u≈439～1492、v≈317～1071；角点凸包仅占全幅约26.1%（这是描述量，不是验收阈值）。因此报告quality=needs_review有根据；不能因RMS低就保证整幅畸变/主点稳定。建议补拍上方、左右边缘、不同远近和左右/上下倾斜，保持整个内角点区域清晰可见。

![采样覆盖与单图误差](采样覆盖与误差.png)

## 棋盘世界位置输入的含义

输入三点：O=(52.2,174.1,−16.9)，U=(52.2,132.1,−16.9)，V=(52.2,174.1,−46.9)mm；OU=42、OV=30且严格垂直，几何格式正确。点选47/40/7对应检测网格两个方向都反转，重排后的O位于原图右下、U左下、V右上，与纸张倒置及所填−Y/−Z方向一致。

但这些数值同时认定：整张棋盘处于**X=52.2平面**，OU严格沿世界−Y，OV严格沿世界−Z。原图能核对点序与方向，不能独立认证这组世界坐标。该图是手持棋盘，仍需确认它实际固定且对齐上述平面/方向，而不是只量一个点后按纸面标称长度假定姿态。角点相对位置给定正确而世界基准偏了，仍可得到低像素RMS。

约2°的摆放倾角，在178mm量级的工作距离上对应约6.2mm侧向量级；仅用于说明灵敏度，不能据此断言本次确实倾斜2°或将差值反调进配置。若U/V实际不与世界轴平行，应填它们真实XYZ，不能强填与O相同的X/Z或X/Y。

## 下一步使用

本次可在网页选“候选0”，为当前任务180°转正流程导出scene_upright外参；未自动应用。建议先将棋盘可靠贴平固定，用另一已知世界摆放位置复测，比较所得C/R是否在独立量测不确定度内一致，并用未参与求解的已知点核验世界定位。镜头/焦点或相机位置改变后不能把这次结果当作旧素材的独立真值。

补拍完善内参后，需要用新K/D重新计算外参，不能把旧候选R/C直接配上新内参就宣称同一标定结果。导轨轴心Y181.5/挡边172.5是导出时附带的既有几何输入，不参与本次棋盘PnP求解，本次并未重新认证挡边9mm关系。

## 证据

输入备份input_intrinsics.json/input_extrinsics.json；summary_initial.json；recalculation.json（逐图误差与诊断对照）；full_resolution_corner_review.json；只读下载fetch_receipt.json及board_evidence/。所有计算仅离线，原始JSON保留不变。未使用/测量此前用户排除的旧照片3/4；当前sample_003/004属于本次独立30张棋盘采样，不是旧滑杆照片。
'''
(out/'标定检查与候选选择.md').write_text(report)
(out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'},ensure_ascii=False,indent=2))
print('Report and coverage chart written.')

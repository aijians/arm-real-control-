from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.colors import HexColor, black, white
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
out=Path(__file__).resolve().parent/'patterns/checkerboard_6mm_8x6_A4.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
c=canvas.Canvas(str(out),pagesize=(210*mm,297*mm))
c.setTitle('6 mm checkerboard - 8 x 6 inner corners - A4 actual size')
c.setAuthor('Camera calibration project')
def text(x,y,t,size=11,color=black):
 c.setFillColor(color);c.setFont('STSong-Light',size);c.drawString(x*mm,y*mm,t)
def line(x1,y1,x2,y2):
 c.setStrokeColor(black);c.setLineWidth(.22*mm);c.line(x1*mm,y1*mm,x2*mm,y2*mm)
text(20,277,'舱内相机标定棋盘',21)
text(20,266,'每格 6 mm   |   8 × 6 内角点   |   A4 矢量原尺寸',12)
text(20,254,'打印：实际大小 / 100%；关闭“适合页面”和缩放。',12)
# Trim card is 70 x 58 mm. Pattern is 54 x 42 mm, centered with 8mm white margin.
x0,y_top=78,232
c.setStrokeColor(HexColor('#777777'));c.setDash(2*mm,2*mm);c.setLineWidth(.15*mm)
c.rect((x0-8)*mm,(y_top-42-8)*mm,70*mm,58*mm,stroke=1,fill=0);c.setDash()
for r in range(7):
 for col in range(9):
  if (r+col)%2==0:
   c.setFillColor(black);c.rect((x0+col*6)*mm,(y_top-(r+1)*6)*mm,6*mm,6*mm,stroke=0,fill=1)
# O/U/V labels stay outside the pattern. No ink is added to detection squares.
text(x0-6,y_top-7,'O',9);line(x0-2,y_top-6,x0-1,y_top-6)
text(x0+56,y_top-7,'U',9);line(x0+55,y_top-6,x0+57,y_top-6)
text(x0-6,y_top-37,'V',9);line(x0-2,y_top-36,x0-1,y_top-36)
text(71,177,'虚线内为裁切卡片：70 × 58 mm',10)
text(20,160,'网页规格：内角点列数 8；行数 6；方格边长填写打印后实测值。',11)
text(20,149,'黑白棋盘本体：54 × 42 mm（9 × 7 个方格）。',11)
text(20,138,'O：靠 O 字样的第一行、第一列内角点，距棋盘左/上边各 6 mm。',10)
text(20,128,'U：与 O 同行、靠 U 字样的最后一个内角点；O→U = 42 mm。',10)
text(20,118,'V：与 O 同列、靠 V 字样的最后一个内角点；O→V = 30 mm。',10)
text(20,106,'外参：输入 O、U、V 三个内角点的世界 XYZ；不是纸边、格心或板中心。',10)
text(20,96,'小格纸板贴平在硬板上，保留白边与 O/U/V 标签；避免弯曲和反光。',10)
text(20,86,'内参采集需改变倾斜、距离和画面位置；外参采集时板与相机保持固定。',10)
text(20,69,'打印尺寸检查（横向与纵向均需核对）',11)
line(25,57,85,57);line(25,55,25,59);line(85,55,85,59);text(43,49,'60 mm 横向',10)
line(160,37,160,67);line(158,37,162,37);line(158,67,162,67);text(165,51,'30 mm',10)
text(20,34,'先量横线 60 mm、竖线 30 mm，再量棋盘总宽 54 mm / 高 42 mm。',10)
text(20,24,'仅打印该页。若尺寸不符，请调整打印比例后重印，并在程序中填实测格距。',10)
c.showPage();c.save()
print(out.resolve())

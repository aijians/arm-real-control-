from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,black,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
p=Path('output/pdf/棋盘OUV内角点标注示意.pdf'); c=canvas.Canvas(str(p),pagesize=(960,850))
c.setTitle('Checkerboard O U V inner-corner guide - not a print target')
def txt(x,y,s,size=18,color=black):
 c.setFont('STSong-Light',size);c.setFillColor(color);c.drawString(x,y,s)
txt(45,800,'O / U / V 三个内角点的位置',28)
txt(45,766,'正对打印纸，文字朝上。彩色圈中心就是要点击、测量的交点。',17)
x,y,s=170,665,65
for r in range(7):
 for col in range(9):
  c.setFillColor(black if (r+col)%2==0 else white)
  c.rect(x+col*s,y-(r+1)*s,s,s,stroke=0,fill=1)
points={'O':(x+s,y-s),'U':(x+8*s,y-s),'V':(x+s,y-6*s)}
colors={'O':'#d92d20','U':'#1570ef','V':'#00875a'}
for k,(px,py) in points.items():
 color=HexColor(colors[k]);c.setStrokeColor(color);c.setLineWidth(3);c.circle(px,py,12,stroke=1,fill=0)
 c.setFillColor(color);c.circle(px,py,3,stroke=0,fill=1)
# Leaders point exactly at the corresponding inner-corner centers.
for k,lx,ly in [('O',66,612),('U',822,612),('V',66,287)]:
 px,py=points[k];color=HexColor(colors[k]);txt(lx,ly,k,32,color)
 c.setStrokeColor(color);c.setLineWidth(2);c.line(lx+25 if k!='U' else lx-8,ly-4,px-14 if k!='U' else px+14,py)
# Dimensions remain outside the board.
c.setStrokeColor(HexColor('#65758b'));c.setLineWidth(1)
c.line(points['O'][0],704,points['U'][0],704)
for px in (points['O'][0],points['U'][0]):c.line(px,697,px,711)
txt(360,716,'O → U：7格 = 42 mm',18)
txt(45,155,'O：第1行、第1列内角点；距黑白棋盘左边、上边各一格（6 mm）。',18)
txt(45,121,'U：第1行、第8列内角点；与 O 同一行，向右跨7格（42 mm）。',18)
txt(45,87,'V：第6行、第1列内角点；与 O 同一列，向下跨5格（30 mm）。',18)
txt(45,40,'本图仅用于认点，请勿打印本图做标定。实物仍使用原6 mm原尺寸棋盘，勿添画彩色圈。',15,HexColor('#b42318'))
c.showPage();c.save();print(p.resolve())

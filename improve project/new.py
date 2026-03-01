import sys,cv2,time,psutil,logging
import numpy as np
from ultralytics import YOLO
from collections import Counter
import winsound

from PyQt5.QtWidgets import QApplication,QLabel,QHBoxLayout,QVBoxLayout,QWidget,QPushButton,QFileDialog,QInputDialog
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QThread,pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logging.getLogger('ultralytics').setLevel(logging.ERROR)

# ================= MODEL =================
model=YOLO("yolov8n.pt")
BASIC_OBJECTS=['person','car','truck','bus','motorcycle','bicycle','traffic light','stop sign']

REAL_HEIGHTS={'person':1.7,'bicycle':1.2,'car':1.5,'motorcycle':1.1,'bus':3.0,'truck':3.5,'traffic light':3.0,'stop sign':2.5}
FOCAL_LENGTH=600

# ⭐ ALERT VARIABLES (ONLY ADDED)
DISTANCE_ALERT_THRESHOLD=5
ALERT_COOLDOWN=2
last_alert_time={}

def estimate_distance(h,label):
    if label not in REAL_HEIGHTS or h<=0:return None
    return round((REAL_HEIGHTS[label]*FOCAL_LENGTH)/h,2)

# ⭐ ALERT FUNCTION (ONLY ADDED)
def play_alert(label):
    global last_alert_time
    t=time.time()
    if label in last_alert_time and (t-last_alert_time[label])<ALERT_COOLDOWN:
        return
    last_alert_time[label]=t
    winsound.Beep(1000,200)

# ================= THREAD =================
class DetectionThread(QThread):
    frame_ready=pyqtSignal(object,object,float,dict)

    def __init__(self,cap):
        super().__init__()
        self.cap=cap
        self.running=True

    def run(self):
        start=time.time()
        count=0
        fps=0

        while self.running and self.cap and self.cap.isOpened():
            ret,frame=self.cap.read()
            if not ret:break

            annotated,counter=self.process_frame(frame)

            count+=1
            if count>=10:
                fps=count/(time.time()-start)
                count=0
                start=time.time()

            self.frame_ready.emit(frame,annotated,fps,counter)

    def process_frame(self,frame):
        results=model(frame,conf=0.4,verbose=False)

        annotated=frame.copy()
        names=results[0].names
        obj_list=[]

        for box in results[0].boxes:
            cls=int(box.cls[0])
            label=names[cls]

            if label not in BASIC_OBJECTS:continue

            obj_list.append(label)
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])

            color=(255,0,0)

            h=y2-y1
            dist=estimate_distance(h,label)

            # ⭐ ALERT TRIGGER (ONLY ADDED)
            if dist and dist<DISTANCE_ALERT_THRESHOLD:
                play_alert(label)
                color=(0,0,255)

            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)

            if dist:
                text=f"{label} {dist}m"
            else:
                text=f"{label} {conf:.2f}"

            cv2.putText(annotated,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        counter=dict(Counter(obj_list))
        return annotated,counter

# ================= DASHBOARD =================
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Analytics Dashboard")
        self.setStyleSheet("background:#0f172a;color:white;")
        self.last_counter={}

        self.video=QLabel()
        self.video.setMinimumSize(820,420)

        self.fps=QLabel("FPS:0")
        self.cpu=QLabel("CPU:0%")
        self.ram=QLabel("RAM:0%")
        self.obj_stats=QLabel("Objects:None")

        self.fig=Figure(figsize=(4,3))
        self.canvas=FigureCanvas(self.fig)

        sidebar=QVBoxLayout()
        sidebar.addWidget(self.btn("Webcam",lambda:self.start(cv2.VideoCapture(0))))
        sidebar.addWidget(self.btn("Mobile",self.mobile))
        sidebar.addWidget(self.btn("Image",self.image))
        sidebar.addWidget(self.btn("Video",self.video_file))
        sidebar.addStretch()

        left_layout=QVBoxLayout()
        left_layout.addLayout(sidebar)
        left_layout.addWidget(self.canvas)

        stat_row=QHBoxLayout()
        stat_row.addWidget(self.fps)
        stat_row.addWidget(self.cpu)
        stat_row.addWidget(self.ram)
        stat_row.addWidget(self.obj_stats)

        main=QVBoxLayout()
        main.addWidget(self.video)
        main.addLayout(stat_row)

        root=QHBoxLayout()
        sw=QWidget()
        sw.setLayout(left_layout)
        root.addWidget(sw,1)
        root.addLayout(main,4)
        self.setLayout(root)

    def btn(self,t,f):
        b=QPushButton(t)
        b.clicked.connect(f)
        b.setStyleSheet("background:#1f2937;color:white;padding:10px;")
        return b

    def start(self,cap):
        self.thread=DetectionThread(cap)
        self.thread.frame_ready.connect(self.update)
        self.thread.start()

    def mobile(self):
        ip,ok=QInputDialog.getText(self,"IP","Enter ip:port")
        if ok:self.start(cv2.VideoCapture(f"http://{ip}/video"))

    def image(self):
        path,_=QFileDialog.getOpenFileName(self)
        if path:
            frame=cv2.imread(path)
            annotated,counter=DetectionThread(None).process_frame(frame)
            self.update(frame,annotated,0,counter)

    def video_file(self):
        path,_=QFileDialog.getOpenFileName(self)
        if path:self.start(cv2.VideoCapture(path))

    def update_graph(self):
        self.fig.clear()
        ax=self.fig.add_subplot(111)
        if self.last_counter:
            labels=list(self.last_counter.keys())
            values=list(self.last_counter.values())
            ax.bar(labels,values)
            ax.set_title("Object Analysis")
        self.canvas.draw()

    def update(self,frame,annotated,fps,counter):
        self.last_counter=counter
        self.update_graph()

        self.fps.setText(f"FPS:{fps:.2f}")
        self.cpu.setText(f"CPU:{psutil.cpu_percent()}%")
        self.ram.setText(f"RAM:{psutil.virtual_memory().percent}%")

        text=" | ".join([f"{k}:{v}" for k,v in counter.items()])
        self.obj_stats.setText("Objects→"+text if text else "Objects:None")

        rgb=cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb.shape
        qt=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qt))

app=QApplication(sys.argv)
w=Dashboard()
w.show()
sys.exit(app.exec_())
from cProfile import label
import sys 
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from skimage.io import imread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "自适应屏幕大小UI"

        #获取显示器分辨率
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()

        print("Screen height {}".format(self.screenheight))
        print("Screen width {}".format(self.screenwidth))

        self.height = int(self.screenheight * 0.5)
        self.width = int(self.screenwidth * 0.5)

        self.resize(self.width,self.height)
        self.wid = QWidget(self)
        self.setCentralWidget(self.wid)
        self.setWindowTitle(self.title)
        self.initUI()

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)

        self.initMenu()
        self.initBrowser()
        self.initImageWindow()

        self.statusBar().showMessage("准备就绪")
        self.wid.setLayout(self.layout)

    def initMenu(self):
        openImageFolderAct = QAction("打开",self)
        openImageFolderAct.setStatusTip("选择一个文件夹，开始标注")
        openImageFolderAct.setShortcut("Ctrl+O")

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&文件')
        fileMenu.addAction(openImageFolderAct)

    def initBrowser(self):
        self.imageBrowser = QListWidget(self)
        self.imageBrowserWidth = int(self.width*0.1)
        self.imageBrowserHeight = int(self.height) 
        self.imageBrowser.setMinimumSize(self.imageBrowserWidth,self.imageBrowserHeight)
        self.layout.addWidget(self.imageBrowser,0,0)

    def initImageWindow(self):
        self.imageWindow = QLabel("Hello World!",self)
        self.imageWindow.setStyleSheet("background-color: darkgray;border: 1px solid black;")
    
        #设置最小的大小，还是以界面大小为基准
        self.imageWindowWidth = int(self.width*0.9)
        self.imageWindowHeight = int(self.height) 
        self.imageWindow.setMinimumSize(self.imageWindowWidth,self.imageWindowHeight)
        #将imageBrowser放置到网格的第0行，第1列
        self.layout.addWidget(self.imageWindow,0,1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


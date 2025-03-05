from PySide6.QtWidgets import QApplication, QMainWindow, QGroupBox, QLabel, QWidget, QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt, QSize
from PySide6.QtGui import QPixmap, QResizeEvent
import qdarkstyle
from PIL.ImageQt import ImageQt
from PIL import Image
from utils import Parameters, WFCProperties, EXTENTIONS
from wfc import WFCSetup, runStep
import sys
import os

# Create the application window
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.app_WFCParams = Parameters()
        self.app_WFCProperties = WFCProperties()

        self.setWindowTitle("WFC")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create all the layouts
        self.layout = QVBoxLayout()
        self.bitmapRequestLayout = QVBoxLayout()
        self.bitmapPathLayout = QHBoxLayout()
        self.bitmapLayout = QGridLayout()
        self.patchSizeLayout = QHBoxLayout()
        self.sizeLayout = QHBoxLayout()
        self.pixelSizeLayout = QHBoxLayout()
        self.resultLayout = QGridLayout()
        self.buttonLayout = QHBoxLayout()
        self.videoLayout = QHBoxLayout()

        # Create a group box for the bitmap and the result parameters
        self.bitmapGroupBox = QGroupBox("Bitmap")
        self.resultGroupBox = QGroupBox("Result")

        # Create all the Widgets

        #### Bitmap Tab ####
        self.textQuery = QLabel(text="Enter the path of the bitmap to use: ")
        self.bitmapPath = QLineEdit()
        self.bitmapSearchButton = QPushButton(text="Open File")
        self.bitmapPath.setText(self.app_WFCParams.input_path)

        self.patchSizeLabel = QLabel(text="Patch Size")
        self.patchSizeSpinBox = QSpinBox()
        self.patchSizeSpinBox.setMinimum(1)
        self.patchSizeSpinBox.setValue(self.app_WFCParams.N)

        self.flipCheckBox = QCheckBox(text="Flip")
        self.flipCheckBox.setChecked(self.app_WFCParams.flip)

        self.rotateCheckBox = QCheckBox(text="Rotate")
        self.rotateCheckBox.setChecked(self.app_WFCParams.rotate)

        self.savePatternsCheckBox = QCheckBox(text="Save Pattern")
        self.savePatternsCheckBox.setChecked(self.app_WFCParams.save_patterns)

        self.printRulesCheckBox = QCheckBox(text="Print Rules")
        self.printRulesCheckBox.setChecked(self.app_WFCParams.print_rules)

        #### Result Tab ####
        self.sizeLabel = QLabel(text="Size")
        self.widthSpinBox = QSpinBox()
        self.widthSpinBox.setMinimum(5)
        self.widthSpinBox.setValue(self.app_WFCParams.width)
        self.sizeDividerLabel = QLabel(text="X")
        self.heightSpinBox = QSpinBox()
        self.heightSpinBox.setMinimum(5)
        self.heightSpinBox.setValue(self.app_WFCParams.height)

        self.pixelSizeLabel = QLabel(text="Pixel Size")
        self.pixelSizeSpinBox = QSpinBox()
        self.pixelSizeSpinBox.setMinimum(1)
        self.pixelSizeSpinBox.setValue(self.app_WFCParams.pixel_size)

        self.recordCheckBox = QCheckBox(text="Record")
        self.recordCheckBox.setChecked(self.app_WFCParams.record)

        self.saveFinalImageCheckBox = QCheckBox(text="Save Final Image")
        self.saveFinalImageCheckBox.setChecked(self.app_WFCParams.save_image)

        self.generateButton = QPushButton(text="Generate")
        self.cancelButton = QPushButton(text="Cancel")
        self.resultLabel = QLabel()

        # Add the widgets to their corresponding layouts

        #### Bitmap Tab ####
        self.bitmapPathLayout.addWidget(self.bitmapPath)
        self.bitmapPathLayout.addWidget(self.bitmapSearchButton)
        self.bitmapRequestLayout.addWidget(self.textQuery)
        self.bitmapRequestLayout.addLayout(self.bitmapPathLayout)
        self.bitmapLayout.addLayout(self.bitmapRequestLayout,0, 0, 1, 2)

        self.patchSizeLayout.addWidget(self.patchSizeLabel)
        self.patchSizeLayout.addWidget(self.patchSizeSpinBox)
        self.bitmapLayout.addLayout(self.patchSizeLayout, 1, 0, 1, 2)

        self.bitmapLayout.addWidget(self.flipCheckBox, 2, 0)
        self.bitmapLayout.addWidget(self.rotateCheckBox, 2, 1)

        self.bitmapLayout.addWidget(self.savePatternsCheckBox, 3, 0)
        self.bitmapLayout.addWidget(self.printRulesCheckBox, 3, 1)

        self.bitmapGroupBox.setLayout(self.bitmapLayout)

        #### Result Tab ####
        self.sizeLayout.addWidget(self.sizeLabel)
        self.sizeLayout.addWidget(self.widthSpinBox)
        self.sizeLayout.addWidget(self.sizeDividerLabel)
        self.sizeLayout.addWidget(self.heightSpinBox)
        self.resultLayout.addLayout(self.sizeLayout, 0, 0, 1, 2)

        self.pixelSizeLayout.addWidget(self.pixelSizeLabel)
        self.pixelSizeLayout.addWidget(self.pixelSizeSpinBox)
        self.resultLayout.addLayout(self.pixelSizeLayout, 1, 0, 1, 2)

        self.resultLayout.addWidget(self.recordCheckBox, 2, 0)
        self.resultLayout.addWidget(self.saveFinalImageCheckBox, 2, 1)

        self.resultGroupBox.setLayout(self.resultLayout)

        #### Button Section ####
        self.buttonLayout.addWidget(self.generateButton)
        self.buttonLayout.addWidget(self.cancelButton)

        #### Video Section ####
        self.videoLayout.addWidget(self.resultLabel)

        #### Global Layout ####
        self.layout.addWidget(self.bitmapGroupBox)
        self.layout.addWidget(self.resultGroupBox)
        self.layout.addLayout(self.buttonLayout)
        self.layout.addLayout(self.videoLayout)

        # Set the main layout properties
        self.layout.addStretch()
        self.layout.setAlignment(Qt.AlignTop)

        # Connect custom function to widget events
        self.bitmapSearchButton.clicked.connect(self.search)

        # Set default properties of the main window
        self.central_widget.setLayout(self.layout)


    # Search bitmap file
    def search(self) -> None:
        bitmapPathName = QFileDialog.getOpenFileName(caption="Open File", dir="/", filter="Bitmap (*.png, *jpg, *jpeg)")
        
        ext = os.path.splitext(bitmapPathName[0])

        if not os.path.exists(bitmapPathName[0]) or ext[1] not in EXTENTIONS:
            return

        self.videoPath.setText(bitmapPathName[0])
        self.app_WFCParams.input_path = bitmapPathName[0]

    # Start ASCIIXEL app in a thread without saving the output
    def clickPreview(self) -> None:
        self.clickCancel()

        # Set asciixel properties and run the setup
        self.app_ASCIIXEL.reset()
        self.app_ASCIIXEL.record = False
        if not self.app_ASCIIXEL.setup(): return

        # Start a new thread
        self.instanced_thread.start()

    # Start ASCIIXEL app in a thread and save the output
    def clickRecord(self) -> None:
        self.clickCancel()

        # Set asciixel properties and run the setup
        self.app_ASCIIXEL.reset()
        self.app_ASCIIXEL.record = True
        if not self.app_ASCIIXEL.setup(): return

        # Start a new thread
        self.instanced_thread.start()
    
    # Cancel the ASCIIXEL app thread
    def clickCancel(self) -> None:
        # Kill thread if it is running
        if self.instanced_thread.isRunning():
            self.instanced_thread.stop()

    # Slot for communicating with the thread worker to get the result image
    @Slot(QPixmap)
    def updateResultImageField(self, img):
        self.resultLabel.setPixmap(img)


# Create signal type
class ImgSignals(QObject):
    signal_img = Signal(QPixmap)

# Create the Worker Thread
class WorkerThread(QThread):
    def __init__(self, parent = None, params: Parameters = None, properties: WFCProperties = None) -> None:
        QThread.__init__(self, parent)

        self.app_WFCParams = params
        self.app_WFCProperties = properties

        self.exit = False

        # Instantiate signals and connect signals to slots
        self.signals = ImgSignals()
        self.signals.signal_img.connect(parent.updateResultImageField)
    
    def run(self) -> None:
        if self.app_WFCParams == None or self.app_WFCProperties == None: return

        while self.app_WFCProperties.status != WAVE_COLLAPSED:
            if self.exit:
                self.exit = False
                return

            self.app_ASCIIXEL.runStep()
            img = QPixmap.fromImage(ImageQt(self.app_ASCIIXEL.out_image))
            self.signals.signal_img.emit(img)

            if self.app_ASCIIXEL.display_original:
                pil_img_orig = Image.fromarray(self.app_ASCIIXEL.cv2_image).convert('RGB')
                img_orig = QPixmap.fromImage(ImageQt(pil_img_orig))
                self.signals.signal_img_orig.emit(img_orig)

        self.app_ASCIIXEL.record_video()
    
    @Slot()
    def stop(self) -> None:
        self.exit = True

if __name__ == '__main__':
#     app = ASCIIXEL(path='videos/Touhou-Bad_Apple!!.mp4', ascii_set=2, display_original=False, reverse_colour=False, output_type=OutputType.PIXEL_ART, record=True)
#     app.run()

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())

    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec())
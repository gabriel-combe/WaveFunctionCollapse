from PySide6.QtWidgets import QApplication, QMainWindow, QGroupBox, QLabel, QWidget, QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt, QSize
from PySide6.QtGui import QPixmap, QResizeEvent
import qdarkstyle
from PIL.ImageQt import ImageQt
from PIL import Image
from utils import Parameters, WFCProperties, EXTENTIONS
from wfc import WFCSetup, runStep, saveResult, RUNNING, CONTRADICTION, WAVE_COLLAPSED
import sys
import os

# Create the application window
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.app_WFCParams = Parameters()

        # Create the Worker Thread Object
        self.instanced_thread = WorkerThread(self)

        self.setWindowTitle("WFC")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create all the layouts
        self.layout = QVBoxLayout()
        self.bitmapRequestLayout = QVBoxLayout()
        self.bitmapPathLayout = QHBoxLayout()
        self.bitmapLayout = QGridLayout()
        self.patchSizeLayout = QHBoxLayout()
        self.outputNameLayout = QVBoxLayout()
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
        self.outputNameLabel = QLabel(text="Output Name")
        self.outputName = QLineEdit()
        self.outputName.setText(self.app_WFCParams.save_pathname)

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
        self.outputNameLayout.addWidget(self.outputNameLabel)
        self.outputNameLayout.addWidget(self.outputName)
        self.resultLayout.addLayout(self.outputNameLayout, 0, 0, 1, 2)

        self.sizeLayout.addWidget(self.sizeLabel)
        self.sizeLayout.addWidget(self.widthSpinBox)
        self.sizeLayout.addWidget(self.sizeDividerLabel)
        self.sizeLayout.addWidget(self.heightSpinBox)
        self.resultLayout.addLayout(self.sizeLayout, 1, 0, 1, 2)

        self.pixelSizeLayout.addWidget(self.pixelSizeLabel)
        self.pixelSizeLayout.addWidget(self.pixelSizeSpinBox)
        self.resultLayout.addLayout(self.pixelSizeLayout, 2, 0, 1, 2)

        self.resultLayout.addWidget(self.recordCheckBox, 3, 0)
        self.resultLayout.addWidget(self.saveFinalImageCheckBox, 3, 1)

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
        self.bitmapPath.textChanged.connect(self.onBitmapPathChanged)
        self.outputName.textChanged.connect(self.onOutputNameChanged)
        self.bitmapSearchButton.clicked.connect(self.search)
        self.patchSizeSpinBox.valueChanged.connect(self.onPatchSizeValueChanged)
        self.flipCheckBox.stateChanged.connect(self.onStateChanged)
        self.rotateCheckBox.stateChanged.connect(self.onStateChanged)
        self.savePatternsCheckBox.stateChanged.connect(self.onStateChanged)
        self.printRulesCheckBox.stateChanged.connect(self.onStateChanged)
        self.widthSpinBox.valueChanged.connect(self.onWidthValueChanged)
        self.heightSpinBox.valueChanged.connect(self.onHeightValueChanged)
        self.pixelSizeSpinBox.valueChanged.connect(self.onPixelSizeValueChanged)
        self.recordCheckBox.stateChanged.connect(self.onStateChanged)
        self.saveFinalImageCheckBox.stateChanged.connect(self.onStateChanged)
        self.generateButton.clicked.connect(self.clickGenerate)
        self.cancelButton.clicked.connect(self.instanced_thread.stop)

        # Set default properties of the main window
        self.central_widget.setLayout(self.layout)


    # Search bitmap file
    def search(self) -> None:
        bitmapPathName = QFileDialog.getOpenFileName(parent=self, caption="Open File", dir='', filter="Bitmaps (*.png *.jpg *.jpeg)")
        
        ext = os.path.splitext(bitmapPathName[0])

        if not os.path.exists(bitmapPathName[0]) or ext[1] not in EXTENTIONS:
            return

        self.bitmapPath.setText(bitmapPathName[0])
        self.app_WFCParams.input_path = bitmapPathName[0]

    def onBitmapPathChanged(self, path: str) -> None:
        self.app_WFCParams.input_path = path

    def onOutputNameChanged(self, path: str) -> None:
        self.app_WFCParams.save_pathname = path

    def onStateChanged(self) -> None:
        self.app_WFCParams.flip = self.flipCheckBox.isChecked()
        self.app_WFCParams.rotate = self.rotateCheckBox.isChecked()
        self.app_WFCParams.save_patterns = self.savePatternsCheckBox.isChecked()
        self.app_WFCParams.print_rules = self.printRulesCheckBox.isChecked()
        self.app_WFCParams.record = self.recordCheckBox.isChecked()
        self.app_WFCParams.save_image = self.saveFinalImageCheckBox.isChecked()

    def onPatchSizeValueChanged(self, value: int) -> None:
        self.app_WFCParams.N = value

    def onWidthValueChanged(self, value: int) -> None:
        self.app_WFCParams.width = value

    def onHeightValueChanged(self, value: int) -> None:
        self.app_WFCParams.height = value

    def onPixelSizeValueChanged(self, value: int) -> None:
        self.app_WFCParams.pixel_size = value

    # Start WFC app in a thread and generate the output
    def clickGenerate(self) -> None:
        self.clickCancel()

        # Set WFC properties and run the setup
        if self.app_WFCParams == None: return
        self.instanced_thread.setup(self.app_WFCParams)

        # Start a new thread
        self.instanced_thread.start()

    # Cancel the WFC app thread
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
    def __init__(self, parent = None, params: Parameters = None) -> None:
        QThread.__init__(self, parent)

        self.app_WFCParams = None
        self.app_WFCProperties = None

        self.exit = False

        # Instantiate signals and connect signals to slots
        self.signals = ImgSignals()
        self.signals.signal_img.connect(parent.updateResultImageField)
    
    def setup(self, params: Parameters = None) -> None:
        self.app_WFCParams = params
        self.app_WFCProperties = WFCSetup(self.app_WFCParams)

        self.exit = False

    def run(self) -> None:
        if self.app_WFCParams == None or self.app_WFCProperties == None: return

        while self.app_WFCProperties.status not in [WAVE_COLLAPSED, CONTRADICTION]:
            if self.exit:
                self.exit = False
                return

            runStep(self.app_WFCParams, self.app_WFCProperties)
            img = QPixmap.fromImage(ImageQt(self.app_WFCProperties.current_img))
            self.signals.signal_img.emit(img)
        
        if self.app_WFCProperties.status == CONTRADICTION: return

        print("The wave has collapsed!")
        saveResult(self.app_WFCParams, self.app_WFCProperties)


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
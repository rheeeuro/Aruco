import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import cv2
import cv2.aruco as aruco
import numpy as np

#
# Aruco
#

class Aruco(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Aruco"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Aruco"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Aruco">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete

#
# ArucoWidget
#

class ArucoWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/Aruco.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = ArucoLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)


    self.initialLayout()
    self.initialTransform()
    self.initialRegistration()
    self.initialConnector()
    self.status = True
    self.ui.startBtn.connect('clicked(bool)', self.onStartBtn)
    self.ui.stopBtn.connect('clicked(bool)', self.onStopBtn)
    self.ui.addBtn.connect('clicked(bool)', self.onAddBtn)
    self.ui.completeBtn.connect('clicked(bool)', self.onCompleteBtn)


  def initialLayout(self):
        # 레이아웃 초기 설정
        defaultLayout = """
          <layout type="horizontal">
           <item>
            <view class="vtkMRMLViewNode" singletontag="1">
             <property name="viewlabel" action="default">1</property>
            </view>
           </item>
           <item>
            <view class="vtkMRMLSliceNode" singletontag="Webcam">
             <property name="orientation" action="default">Sagittal</property>
             <property name="viewlabel" action="default">W</property>
             <property name="viewcolor" action="default">#333333</property>
             <property name="viewgroup" action="default\">2</property>
            </view>
           </item>
          </layout>
        
        """
        self.layoutManager = slicer.app.layoutManager()
        self.layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(501, defaultLayout)
        # Switch to the new custom layout
        self.layoutManager.setLayout(501)

  def initialTransform(self):
    self.arucoWebcam = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    self.arucoWebcam.SetName('arucoWebcam')

    self.webcamToSlicer = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    self.webcamToSlicer.SetName('webcamToSlicer')
    webcamToSlicerMatrix = vtk.vtkMatrix4x4()
    webcamToSlicerMatrix.SetElement(0, 0, -1)
    webcamToSlicerMatrix.SetElement(1, 1, 0)
    webcamToSlicerMatrix.SetElement(2, 2, 0)
    webcamToSlicerMatrix.SetElement(1, 2, 1)
    webcamToSlicerMatrix.SetElement(2, 1, 1)
    self.webcamToSlicer.SetMatrixTransformToParent(webcamToSlicerMatrix)

    self.p = slicer.modules.createmodels.logic().CreateNeedle(80.0, 1.0, 2.5, False, None)
    self.p.SetAndObserveTransformNodeID(self.arucoWebcam.GetID())

    self.webcamFix = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    self.webcamFix.SetName('webcamFix')

    self.vertorFix = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    self.vertorFix.SetName('vertorFix')
    vertorFixMatrix = vtk.vtkMatrix4x4()
    vertorFixMatrix.SetElement(0, 0, -1)
    vertorFixMatrix.SetElement(1, 1, 0)
    vertorFixMatrix.SetElement(2, 2, 0)
    vertorFixMatrix.SetElement(1, 2, 1)
    vertorFixMatrix.SetElement(2, 1, 1)
    self.vertorFix.SetMatrixTransformToParent(webcamToSlicerMatrix)


  def initialConnector(self):
    # IGTLConnector 노드 생성
    self.optiTrackConnector = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLIGTLConnectorNode')

    # 서버 포트 설정 (OptiTrack - 18944, NDI Aurora - 18945)
    self.optiTrackConnector.SetServerPort(18944)
    # Connector 시작
    self.optiTrackConnector.Start()

  def initialRegistration(self):
    # Fiducial Registration Wizard, 노드 생성 및 연결
    self.fiducialRegistrationWizard = slicer.mrmlScene.AddNewNodeByClass(
        'vtkMRMLFiducialRegistrationWizardNode')
    self.fiducialRegistrationWizard.SetName('FiducialRegWizard')

    self.fiducialReg = slicer.mrmlScene.AddNewNodeByClass(
      'vtkMRMLLinearTransformNode')
    self.fiducialReg.SetName('FiducialReg')

    self.fromFids = slicer.mrmlScene.AddNewNodeByClass(
        'vtkMRMLMarkupsFiducialNode')
    self.fromFids.SetName('From')
    self.fromFids.SetDisplayVisibility(False)
    self.toFids = slicer.mrmlScene.AddNewNodeByClass(
        'vtkMRMLMarkupsFiducialNode')
    self.toFids.SetName('To')
    self.toFids.SetMarkupLabelFormat('%d')
    self.toFids.GetDisplayNode().SetSelectedColor(1, 1, 0)

    self.fiducialRegistrationWizard.SetAndObserveFromFiducialListNodeId(
        self.fromFids.GetID())
    self.fiducialRegistrationWizard.SetAndObserveToFiducialListNodeId(
        self.toFids.GetID())
    self.fiducialRegistrationWizard.SetOutputTransformNodeId(
        self.fiducialReg.GetID())

  def onStartBtn(self):
    self.videoCapture()

  def onStopBtn(self):
    self.status = False
    self.optiTrackWebcam.RemoveObserver(self.onTransformModifiedTag)

  def videoCapture(self):
    cap = cv2.VideoCapture(1)
    filepath = os.path.abspath(os.getcwd()).replace(
            '\\', '/') + '/Aruco/Aruco/calibrationCoefficients.yaml'
    cv_file = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    matrix_coefficients = cv_file.getNode("camera_matrix").mat()
    distortion_coefficients = cv_file.getNode("dist_coeff").mat()

    print(matrix_coefficients)
    print(distortion_coefficients)

    cv_file.release()
    self.volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    self.volumeNode.CreateDefaultDisplayNodes()

    cx = matrix_coefficients[0][2]
    cy = matrix_coefficients[1][2]
    fx = matrix_coefficients[0][0]
    fy = matrix_coefficients[1][1]

    markerModel = slicer.modules.createmodels.logic().CreateCube(100, 0.1, 100, None)

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(13)
    points.SetPoint(0,  0.0, 0.0, 0.0)
    points.SetPoint(1,   cx, cy, fx)
    points.SetPoint(2,   cx, -cy, fx)
    points.SetPoint(3,   0.0,  0.0, 0.0)
    points.SetPoint(4,   -cx, cy, fx)
    points.SetPoint(5,   -cx, -cy, fx)
    points.SetPoint(6,   0.0,  0.0, 0.0)
    points.SetPoint(7,   cx, cy, fx)
    points.SetPoint(8,   -cx, cy, fx)
    points.SetPoint(9,   0.0,  0.0, 0.0)
    points.SetPoint(10,   cx, -cy, fx)
    points.SetPoint(11,   -cx, -cy, fx)
    points.SetPoint(12,   0.0,  0.0, 0.0)
    line = vtk.vtkLineSource()
    line.SetPoints(points)
    self.lineNode = slicer.modules.models.logic().AddModel(line.GetOutputPort())
    self.lineNode.SetAndObserveTransformNodeID(self.arucoWebcam.GetID())

    webcamFixMatrix = vtk.vtkMatrix4x4()
    webcamFixMatrix.SetElement(0, 0, 0)
    webcamFixMatrix.SetElement(1, 1, 0)
    webcamFixMatrix.SetElement(2, 2, 0)
    webcamFixMatrix.SetElement(0, 1, 1)
    webcamFixMatrix.SetElement(1, 2, 1)
    webcamFixMatrix.SetElement(2, 0, -1)

    webcamFixMatrix.SetElement(0, 3, -1*cx)
    webcamFixMatrix.SetElement(1, 3, -1*cy)
    webcamFixMatrix.SetElement(2, 3, fx)
    self.webcamFix.SetMatrixTransformToParent(webcamFixMatrix)

    def intervalFunction():
      success, img = cap.read()
      arucoPosition = self.findArucoMarkers(img, matrix_coefficients, distortion_coefficients)

      for p in arucoPosition:
        cv2.circle(img,(int(p[0]), int(p[1])), 5, (255, 0, 0), -1)

      slicer.util.updateVolumeFromArray(self.volumeNode, img)
      slicer.util.setSliceViewerLayers(background=self.volumeNode)
      self.volumeNode.SetAndObserveTransformNodeID(self.webcamFix.GetID())
      slicer.util.resetSliceViews()
      self.webcamFix.SetAndObserveTransformNodeID(self.arucoWebcam.GetID())

      slicer.modules.volumereslicedriver.logic().SetDriverForSlice(self.volumeNode.GetID(), slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeWebcam'))
      slicer.modules.volumereslicedriver.logic().SetModeForSlice(4, slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeWebcam'))

      # cv2.imshow("Image", img)        
      # cv2.waitKey(1)

    self.timer = qt.QTimer()
    self.timer.setInterval(1)
    self.timer.start()
    self.timer.timeout.connect(intervalFunction)

  def findArucoMarkers(self, img, matrix_coefficients, distortion_coefficients, markerSize=4, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)
    if draw:
      aruco.drawDetectedMarkers(img, corners)


    cx = matrix_coefficients[0][2]
    cy = matrix_coefficients[1][2]
    fx = matrix_coefficients[0][0]
    fy = matrix_coefficients[1][1]

    arucoPosition = []

    if np.all(ids is not None):
      zipped = zip(ids, corners)
      ids, corners = zip(*(sorted(zipped)))
      for i in range(0, len(ids)):  # Iterate in markers
      # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 100, matrix_coefficients, distortion_coefficients)
        
        # https://github.com/naruya/aruco
        R = cv2.Rodrigues(rvec)[0]
        R_T = R.T
        T = tvec[0].T

        xyz = np.dot(R_T, - T).squeeze()
        rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])

        cameraMatrix = vtk.vtkMatrix4x4()
        for i in range(3):
          for j in range(3):
            cameraMatrix.SetElement(i, j, R_T[i][j])
        for i in range(3):
          cameraMatrix.SetElement(i, 3, xyz[i])    
        self.arucoWebcam.SetMatrixTransformToParent(cameraMatrix)
 

        x = (fx * tvec[0][0][0] / tvec[0][0][2]) + cx
        y = (fy * tvec[0][0][1] / tvec[0][0][2]) + cy
        arucoPosition.append([x, y])

        # self.arucoWebcam.SetAndObserveTransformNodeID(self.webcamToSlicer.GetID())
        self.arucoWebcam.SetAndObserveTransformNodeID(self.webcamToSlicer.GetID())


    return arucoPosition


  def onAddBtn(self):
    if (self.optiTrackConnector.GetNumberOfIncomingMRMLNodes() < 2):
      self.needleToTracker = None
      self.webcamToTracker = None
      return
    else:
      for i in range(self.optiTrackConnector.GetNumberOfIncomingMRMLNodes()):
        node = self.optiTrackConnector.GetIncomingMRMLNode(i)
        if node.GetName() == 'NeedleToTracker':
            self.needleToTracker = node
        if node.GetName() == 'WebcamToTracker':
            self.webcamToTracker = node

      slicer.modules.fiducialregistrationwizard.logic().AddFiducial(self.webcamToTracker, self.fromFids)
      slicer.modules.fiducialregistrationwizard.logic().AddFiducial(self.arucoWebcam, self.toFids)
    

  def onCompleteBtn(self):
    slicer.modules.fiducialregistrationwizard.logic().UpdateCalibration(self.fiducialRegistrationWizard)
    for i in range(self.fromFids.GetNumberOfFiducials()):
      self.fromFids.SetNthFiducialVisibility(i, False)
      self.toFids.SetNthFiducialVisibility(i, False)

    self.vertorFix.SetAndObserveTransformNodeID(self.webcamToTracker.GetID())
    self.webcamToTracker.SetAndObserveTransformNodeID(self.fiducialReg.GetID())
    

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed


  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()



#
# ArucoLogic
#

class ArucoLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

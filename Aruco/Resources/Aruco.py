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
# Acuro
#

class Acuro(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Acuro"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Acuro"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Acuro">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete

#
# AcuroWidget
#

class AcuroWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/Acuro.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = AcuroLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)



    self.ui.startBtn.connect('clicked(bool)', self.onStartBtn)



  def onStartBtn(self):
    cap = cv2.VideoCapture(1)

    cv_file = cv2.FileStorage("calibrationCoefficients.yaml", cv2.FILE_STORAGE_READ)
   
    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    matrix_coefficients = cv_file.getNode("camera_matrix").mat()
    distortion_coefficients = cv_file.getNode("dist_coeff").mat()

    print(matrix_coefficients)
    print(distortion_coefficients)

    cv_file.release()


    while True:
        success, img = cap.read()
        arucoPosition = self.findArucoMarkers(img, matrix_coefficients, distortion_coefficients)

        for p in arucoPosition:
            cv2.circle(img,(int(p[0]), int(p[1])), 5, (255, 0, 0), -1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

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
        xyz = tvec[0][0]

        x = (fx * xyz[0] / xyz[2]) + cx
        y = (fy * xyz[1] / xyz[2]) + cy

        arucoPosition.append([x, y])
        print(f'{ids[i][0]}: {tvec[0][0]}')
        print()

    return arucoPosition

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
# AcuroLogic
#

class AcuroLogic(ScriptedLoadableModuleLogic):
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

import os, unittest, tempfile
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLabel
from annotation import Annotation, AnnotationDialog, AnnotationCanvas
from main import DEFAULT_SETTINGS, _validate_settings
from shortcuts import DEFAULT_SHORTCUTS
from training_dialog import TrainingDialog
from settings_dialog import SettingsDialog
from guide_dialog import GuideDialog
from modeling import training_model_path
from naming import numbered_output

class UpdateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.app=QApplication.instance() or QApplication([])
 def tearDown(self):
  for w in QApplication.topLevelWidgets():
   w.hide(); w.deleteLater()
  QApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
 def test_model_choices_and_validation(self):
  with tempfile.TemporaryDirectory() as temp:
   d=TrainingDialog(Path(temp));d.data_edit.setText("data.yaml")
   self.assertEqual(d.current_config()["model_family"], "yolo26")
   for family in ("yolo26","yolo11","yolov8"):
    for task, suffix in (("detect",""),("pose","-pose"),("segment","-seg"),("obb","-obb")):
     self.assertEqual(str(training_model_path(family,"n",task)),f"{family}n{suffix}.pt")
   d.model_combo.setCurrentIndex(3)
   with self.assertRaises(ValueError):d.current_config()
 def test_number_reservation_preserves_existing_and_handles_concurrent_writer(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp); (root/"labels").mkdir(); (root/"labels/sample_000008.txt").write_text("keep")
   with numbered_output(root,"sample_",("images","labels","meta")) as first:
    with numbered_output(root,"sample_",("images","labels","meta")) as second:
     self.assertEqual(first,"sample_000009");self.assertEqual(second,"sample_000010")
   self.assertEqual((root/"labels/sample_000008.txt").read_text(),"keep")
   self.assertFalse(list(root.glob("*.lock")))
 def test_custom_canvas_keys_replace_defaults(self):
  canvas=AnnotationCanvas(np.zeros((200,200,3),np.uint8),[Annotation([50,50,100,100])],np.zeros((4,2),np.float32))
  canvas.selected_box=0;canvas.shortcuts={"delete":"F8","move_right":"F7"}
  canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress,Qt.Key.Key_Right,Qt.KeyboardModifier.NoModifier))
  self.assertEqual(canvas.annotations[0].box[0],50)
  canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress,Qt.Key.Key_F7,Qt.KeyboardModifier.NoModifier))
  self.assertEqual(canvas.annotations[0].box[0],51)
  canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress,Qt.Key.Key_Delete,Qt.KeyboardModifier.NoModifier))
  self.assertEqual(len(canvas.annotations),1)
  canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress,Qt.Key.Key_F8,Qt.KeyboardModifier.NoModifier))
  self.assertEqual(len(canvas.annotations),0)
 def test_hit_margin_is_five_screen_pixels_at_every_zoom(self):
  canvas=AnnotationCanvas(np.zeros((200,200,3),np.uint8),[],np.zeros((4,2),np.float32))
  canvas.resize(800,600)
  for zoom in (0.5,1,3):
   canvas._zoom=zoom
   scale=abs(canvas._image_to_widget(1,0).x()-canvas._image_to_widget(0,0).x())
   for angle in (0,30,90):
    box=Annotation([50,50,100,100],angle=angle)
    from geometry import rotated_box_points
    corners=rotated_box_points(box.box,angle)
    mid=(corners[0]+corners[1])/2
    normal=mid-np.array([75,75]);normal/=np.linalg.norm(normal)
    self.assertTrue(canvas._contains(box,tuple(mid+normal*4/scale)))
    self.assertFalse(canvas._contains(box,tuple(mid+normal*7/scale)))
 def test_shortcuts_persist_and_conflicts_are_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   d=SettingsDialog(DEFAULT_SETTINGS,Path(temp))
   self.assertEqual(set(d.shortcut_edits),set(DEFAULT_SHORTCUTS))
   d.shortcut_edits["undo"].setText("F6")
   self.assertEqual(_validate_settings(d.values())["shortcuts"]["undo"],"F6")
   d.shortcut_edits["delete"].setText("F6")
   with self.assertRaises(ValueError):d.values()
 def test_guide_contains_clickable_versions_link(self):
  d=GuideDialog(Path("missing.jpg"))
  links=[x for x in d.findChildren(QLabel) if "github.com/lukkad612/LabelingCapture/tags" in x.text()]
  self.assertEqual(len(links),1);self.assertTrue(links[0].openExternalLinks())

if __name__=="__main__":unittest.main()

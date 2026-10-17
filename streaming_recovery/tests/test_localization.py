"""局部图像定位与腕部候选回归；真实 JPEG 检查单列，不代表泛化实验。"""
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from streaming_recovery.__main__ import load_config
from streaming_recovery.contracts import Observation
from streaming_recovery.localization import localize_wrist_handle_and_fingers, image_alignment_directions
from streaming_recovery.multiview import MultiViewGeometryVision

REAL=Path(__file__).resolve().parents[1]/'artifacts/multimodal_20261017/final_run'

class LocalizationTests(unittest.TestCase):
    def test_real_visible_blade_segment_survives_lower_reflection(self):
        cfg=load_config();vision=MultiViewGeometryVision(cfg['geometry'],cfg['rules'])
        rows=[]
        for t in (5,6,7,8):
            p=REAL/f'trajectory_000019/frames/obs-{t*25:06d}_cam_right_wrist.jpg'
            if not p.exists():self.skipTest('verified local real image unavailable')
            obs=Observation(float(t),'',{'cam_right_wrist':Image.open(p)},None,{},f'obs-{t*25:06d}')
            rows.append(vision.wrist_observation(obs))
        self.assertTrue(all(r['blade'] is not None and r['joined'] for r in rows),rows)
        self.assertTrue(any(r['cut_operation'] for r in rows[1:]),rows)

    def test_real_handle_neck_and_two_finger_edges(self):
        p=REAL/'trajectory_000002/frames/obs-000050_cam_right_wrist.jpg'
        if not p.exists():self.skipTest('verified local real image unavailable')
        r=localize_wrist_handle_and_fingers(Image.open(p))
        self.assertEqual(r['status'],'localized')
        self.assertEqual(len(r['finger_inner_points_pixels']),2)
        self.assertEqual(len(r['handle_neck_pixels']['edge_segments']),5)
        self.assertEqual(image_alignment_directions(r['knife_handle_xy'],r['gripper_xy'],50),['图像左侧'])
        self.assertNotIn('grasp_readiness',r)

    def test_real_merged_finger_and_holder_rejected(self):
        p=REAL/'trajectory_000008/frames/obs-000100_cam_right_wrist.jpg'
        if not p.exists():self.skipTest('verified local real image unavailable')
        r=localize_wrist_handle_and_fingers(Image.open(p))
        self.assertEqual(r['status'],'unknown')
        self.assertIsNone(r['knife_handle_xy']);self.assertIsNone(r['gripper_xy'])

    def test_textureless_images_cannot_supply_positions(self):
        for gray in (0,128,255):
            r=localize_wrist_handle_and_fingers(np.full((384,512,3),gray,dtype='uint8'))
            self.assertEqual(r['status'],'unknown');self.assertIsNone(r['knife_handle_xy'])

    def test_direction_uses_image_axes_and_deadband_only(self):
        self.assertEqual(image_alignment_directions([400,800],[600,600],50),['图像左侧','图像下方'])
        self.assertEqual(image_alignment_directions([550,550],[500,500],50),[])

if __name__=='__main__':unittest.main()

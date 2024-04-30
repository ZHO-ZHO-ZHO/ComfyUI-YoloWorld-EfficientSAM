import unittest
from unittest.mock import patch, MagicMock
import sys

# Mocking the modules that YOLO_WORLD_EfficientSAM.py imports
mocked_modules = {
    'folder_paths': MagicMock(),
    'cv2': MagicMock(),
    'numpy': MagicMock(),
    'torch': MagicMock(),
    'supervision': MagicMock(),
    'tqdm': MagicMock(),
    'inference.models': MagicMock(),
    'YOLO_WORLD_EfficientSAM.utils.efficient_sam': MagicMock(),
    'YOLO_WORLD_EfficientSAM.utils.video': MagicMock(),
    'YOLO_WORLD_EfficientSAM.utils': MagicMock(),
}
sys.modules.update(mocked_modules)

# Loading the process_categories function directly from the file
with open('YOLO_WORLD_EfficientSAM.py') as f:
    code = compile(f.read(), 'YOLO_WORLD_EfficientSAM.py', 'exec')
    exec_namespace = {'__name__': '__main__'}
    exec(code, exec_namespace)
process_categories = exec_namespace['process_categories']

class TestYOLOWORLDEfficientSAM(unittest.TestCase):

    def test_process_categories(self):
        self.assertEqual(process_categories("cat,dog,bird"), ["cat", "dog", "bird"])
        self.assertEqual(process_categories("cat, dog , bird "), ["cat", "dog", "bird"])
        self.assertEqual(process_categories(""), [])

if __name__ == '__main__':
    unittest.main()

import unittest

def mock_process_categories(categories):
    return [category.strip() for category in categories.split(',') if category.strip()]

class TestProcessCategories(unittest.TestCase):
    def test_process_categories(self):
        self.assertEqual(mock_process_categories("cat,dog,bird"), ["cat", "dog", "bird"])
        self.assertEqual(mock_process_categories("cat, dog , bird "), ["cat", "dog", "bird"])
        self.assertEqual(mock_process_categories(""), [])
        self.assertEqual(mock_process_categories(" , "), [])

if __name__ == '__main__':
    unittest.main()

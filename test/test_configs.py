# tests/test_configs.py

import unittest
from src.configs import load_config

class TestConfigs(unittest.TestCase):
    def test_load_config(self):
        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn('GENERAL', config)
        self.assertIn('RANGES', config)
        # Añade más aserciones según tu configuración

if __name__ == '__main__':
    unittest.main()

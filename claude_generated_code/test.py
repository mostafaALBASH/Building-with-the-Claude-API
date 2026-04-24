import unittest
from main import calculate_pi


class TestCalculatePi(unittest.TestCase):

    def test_pi_default_5_digits(self):
        """Test that pi is correctly calculated to the 5th digit by default."""
        result = calculate_pi()
        self.assertEqual(result, 3.14159)

    def test_pi_rounded_to_5_digits(self):
        """Test that the result is rounded to exactly 5 decimal places."""
        result = calculate_pi(5)
        decimal_places = len(str(result).split(".")[1])
        self.assertLessEqual(decimal_places, 5)

    def test_pi_is_float(self):
        """Test that the return type is a float."""
        result = calculate_pi()
        self.assertIsInstance(result, float)

    def test_pi_custom_digits(self):
        """Test pi calculated to fewer decimal places."""
        self.assertEqual(calculate_pi(0), 3.0)
        self.assertEqual(calculate_pi(1), 3.1)
        self.assertEqual(calculate_pi(2), 3.14)

    def test_pi_within_tolerance(self):
        """Test that the result is within an acceptable tolerance of the true value."""
        import math
        result = calculate_pi(5)
        self.assertAlmostEqual(result, math.pi, places=5)


if __name__ == "__main__":
    unittest.main()

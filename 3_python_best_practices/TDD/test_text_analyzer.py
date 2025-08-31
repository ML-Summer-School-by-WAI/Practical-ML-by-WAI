
import unittest
from text_analyzer import TextAnalyzer

class TestTextAnalyzer(unittest.TestCase):

    class TestTextAnalyzer(unittest.TestCase):
    
    # Let's use the setUp method to avoid repeating code
        def setUp(self):
            """This method runs before each test."""
            self.simple_analyzer = TextAnalyzer("hello")
            self.sentence_analyzer = TextAnalyzer("Hello world this is a test")
            self.empty_analyzer = TextAnalyzer("") # New analyzer for edge cases

            

        def test_char_count(self):
            self.assertEqual(self.simple_analyzer.char_count(), 5)
            self.assertEqual(self.empty_analyzer.char_count(), 0) # Add assertion here

        def test_word_count(self):
            self.assertEqual(self.sentence_analyzer.word_count(), 6)
            self.assertEqual(self.empty_analyzer.word_count(), 0) # Add assertion here


if __name__ == '__main__':
    unittest.main()
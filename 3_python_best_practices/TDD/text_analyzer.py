class TextAnalyzer:
    def __init__(self, text):
        self.text = text
    
    def char_count(self):
        # The simplest way to make the test pass is len()
        return len(self.text)
    

     # NEW METHOD!
    def word_count(self):
        # The simplest way is to split the text by spaces and count the parts.
        words = self.text.split()
        return len(words)
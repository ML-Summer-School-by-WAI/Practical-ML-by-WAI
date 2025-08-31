import pytest
import os
import re

# Define the path to the marks file from the root of the repository
MARKS_FILE_PATH = "6_deep_learning/CNN/tf_best_practices/binary_classification/marks.txt"

def get_mark():
    """Helper function to read the mark from the file."""
    if not os.path.exists(MARKS_FILE_PATH):
        pytest.fail(f"Required file not found: {MARKS_FILE_PATH}")
    try:
        with open(MARKS_FILE_PATH, 'r') as f:
            content = f.read().strip()
            match = re.search(r'(\d+)', content)
            if match:
                return int(match.group(1))
            else:
                pytest.fail(f"No valid number found in {MARKS_FILE_PATH}")
    except Exception as e:
        pytest.fail(f"Could not read a valid integer mark from the file. Error: {e}")

@pytest.fixture(scope="session")
def mark():
    """A pytest fixture to provide the mark to all tests efficiently."""
    return get_mark()

def test_passed_40(mark):
    """Checks if mark is greater than 40."""
    assert mark > 40

def test_passed_50(mark):
    """Checks if mark is greater than 50."""
    assert mark > 50

def test_passed_60(mark):
    """Checks if mark is greater than 60."""
    assert mark > 60

def test_passed_70(mark):
    """Checks if mark is greater than 80."""
    assert mark > 70

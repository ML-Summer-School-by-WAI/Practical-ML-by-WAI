### The TDD Mantra: Red, Green, Refactor
TDD is a cycle built on three simple steps. You repeat this cycle for every tiny piece of functionality you add.
- RED: Write a failing test. Write a small test for a feature that doesn't exist yet. Since the feature isn't there, the test must fail (turn red). This proves your test is working correctly and that the feature is actually needed.
- GREEN: Write the simplest possible code to make the test pass. Don't over-engineer it. Don't add extra features. Your only goal is to make that one specific test pass (turn green). This might even feel a bit "dumb" or simplistic, and that's okay.
- REFACTOR: Clean up the code while keeping the test green. Now that you have a working feature and a safety net (the test), you can improve your code's design, remove duplication, or make it more readable, all while re-running the test to ensure you haven't broken anything.


```bash
python -m unittest test_text_analyzer.py
```
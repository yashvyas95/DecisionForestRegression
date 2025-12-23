# Contributing to Decision Forest Regression

Thank you for your interest in contributing to this project! We welcome contributions of all types, from bug reports to feature implementations. This document provides guidelines to help you contribute effectively.

## Code of Conduct

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive feedback
- Report unacceptable behavior to the project maintainer

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with machine learning concepts
- Understanding of ensemble learning methods

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/DecisionForestRegression.git
   cd DecisionForestRegression
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 mypy
   ```

5. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We follow PEP 8 with some modifications:

- Line length: 100 characters (soft limit)
- Use double quotes for strings
- Use type hints for all functions

Format your code before committing:
```bash
black src/ tests/
flake8 src/ tests/ --max-line-length=100
mypy src/ --ignore-missing-imports
```

### Type Hints

All functions should include type hints:

```python
from typing import List, Dict, Optional
import numpy as np

def engineer_features(
    X: np.ndarray,
    polynomial_degree: int = 2
) -> np.ndarray:
    """
    Engineer features from raw input data.
    
    Args:
        X: Input features of shape (n_samples, n_features)
        polynomial_degree: Degree of polynomial features
        
    Returns:
        Engineered features of shape (n_samples, n_engineered_features)
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_feature_importance(
    forest: DecisionForest,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Calculate feature importance using permutation method.
    
    Args:
        forest: Trained DecisionForest instance
        X: Feature matrix
        y: Target values
        
    Returns:
        Dictionary mapping feature names to importance scores
        
    Raises:
        ValueError: If X and y have mismatched shapes
        
    Example:
        >>> forest = DecisionForest(n_trees=50)
        >>> forest.fit(X_train, y_train)
        >>> importance = calculate_feature_importance(forest, X_test, y_test)
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_decision_forest.py -v

# Run specific test
pytest tests/unit/test_decision_forest.py::TestDecisionForest::test_fit -v
```

### Writing Tests

Create tests in `tests/` directory matching source structure:

```python
import pytest
import numpy as np
from src.decision_forest.core.forest import DecisionForest

class TestDecisionForest:
    """Test suite for DecisionForest implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        return X, y
    
    def test_forest_initialization(self):
        """Test DecisionForest can be initialized."""
        forest = DecisionForest(n_trees=10)
        assert forest.n_trees == 10
    
    def test_fit_predict(self, sample_data):
        """Test forest can fit data and make predictions."""
        X, y = sample_data
        forest = DecisionForest(n_trees=5)
        forest.fit(X, y)
        
        predictions = forest.predict(X)
        assert predictions.shape == (100,)
```

### Test Coverage

Aim for:
- Minimum 80% line coverage
- All public methods tested
- Edge cases and error conditions tested
- Integration tests for API endpoints

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add feature importance calculation

Implement permutation-based feature importance to help users
understand model decisions. Adds calculate_feature_importance()
function to metrics module.

Fixes #42
```

Commit message format:
- Type: feat, fix, docs, style, refactor, test, chore
- Subject: Brief description (50 chars max)
- Body: Detailed explanation (wrap at 72 chars)
- Footer: Reference issues (Fixes #123)

### Pull Request Process

1. Ensure all tests pass and code is properly formatted
2. Create a pull request with a clear description
3. Reference related issues in the PR description
4. Ensure PR title follows: `[Type] Short description`
5. Request review from maintainers

PR Description Template:

```markdown
## Description
Brief explanation of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## How to Test
Steps to verify the changes work correctly

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Areas for Contribution

### High Priority

- [ ] SHAP-based model explanation
- [ ] Batch prediction optimization
- [ ] Multi-region model support
- [ ] GPU acceleration for tree building

### Medium Priority

- [ ] Hyperparameter optimization framework
- [ ] Advanced visualization tools
- [ ] API rate limiting and caching
- [ ] Comprehensive benchmarking suite

### Good for Beginners

- [ ] Documentation improvements
- [ ] Example notebooks
- [ ] Additional unit tests
- [ ] Code comments and docstrings

## Reporting Issues

### Bug Reports

Include:
1. Python version and OS
2. Steps to reproduce
3. Expected vs actual behavior
4. Error traceback (if applicable)
5. Minimal code example

Template:
```markdown
## Description
What's the bug?

## Environment
- Python version: 3.9
- OS: Windows 10
- Package versions: [output of pip freeze]

## Steps to Reproduce
1. ...
2. ...

## Expected Behavior
What should happen?

## Actual Behavior
What actually happens?

## Additional Context
Screenshots, logs, or other details
```

### Feature Requests

Include:
1. Clear description of desired feature
2. Use case and motivation
3. Potential implementation approaches
4. Related issues or PRs

## Documentation

### README Updates

Update README.md when adding:
- New features
- Different usage patterns
- Architecture changes
- Performance improvements

### Inline Documentation

All functions should have docstrings explaining:
- Purpose and behavior
- Parameters and return types
- Exceptions that can be raised
- Usage examples for complex functions

## Performance Considerations

When adding features, ensure:

1. Benchmark impact on inference speed
2. Document memory requirements
3. Test with large datasets (10K+ samples)
4. Avoid unnecessary copies/allocations

Benchmark example:
```python
import time
import numpy as np
from src.decision_forest.core.forest import DecisionForest

X = np.random.randn(10000, 14)
y = np.random.randn(10000)

forest = DecisionForest(n_trees=50)
forest.fit(X, y)

start = time.time()
predictions = forest.predict(X)
elapsed = time.time() - start

print(f"Prediction time: {elapsed:.3f}s ({len(X)/elapsed:.0f} samples/s)")
```

## Questions?

- Check existing issues and discussions
- Review project documentation
- Open a discussion thread
- Contact maintainers

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for making this project better!

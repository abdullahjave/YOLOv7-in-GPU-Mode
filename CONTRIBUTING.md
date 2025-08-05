# Contributing to YOLOv7 GPU Mode

First off, thank you for considering contributing to YOLOv7 GPU Mode! 🎉

It's people like you that make YOLOv7 GPU Mode such a great tool for the computer vision community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

### Types of Contributions

We welcome many different types of contributions:

🐛 **Bug Reports**: Help us identify and fix bugs
📖 **Documentation**: Improve or add documentation
✨ **Features**: Propose and implement new features
🎨 **Examples**: Add usage examples and tutorials
⚡ **Performance**: Optimize code for better GPU utilization
🧪 **Testing**: Improve test coverage
🔧 **Tools**: Develop helpful utilities and scripts

## How to Contribute

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

When reporting bugs, please include:
- Operating system and version
- Python version
- PyTorch and CUDA versions
- GPU model and driver version
- Exact error message and stack trace
- Minimal code to reproduce the issue
- Expected vs actual behavior

### 💡 Suggesting Features

Feature suggestions are welcome! Please:
- Check if the feature already exists or is planned
- Clearly describe the use case and benefits
- Provide examples of how it would be used
- Consider GPU performance implications

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/YOLOv7-in-GPU-Mode.git
cd YOLOv7-in-GPU-Mode
```

### 2. Set Up Environment

```bash
# Create development environment
conda create -n yolov7-dev python=3.8
conda activate yolov7-dev

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements_gpu.txt

# Install development dependencies
pip install black flake8 pytest pytest-cov
```

### 3. Create Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

## Pull Request Process

### 1. Make Changes

- Keep changes focused and atomic
- Write clear, descriptive commit messages
- Test your changes thoroughly
- Update documentation as needed

### 2. Code Quality

```bash
# Format code
black .

# Check linting
flake8 .

# Run tests
pytest tests/
```

### 3. Commit Guidelines

Use descriptive commit messages:

```bash
# Good examples
git commit -m "🐛 Fix GPU memory leak in batch processing"
git commit -m "✨ Add support for custom anchor sizes"
git commit -m "📖 Update installation guide for Windows"
git commit -m "⚡ Optimize dataloader for multi-GPU training"

# Use these prefixes:
# 🐛 :bug: for bug fixes
# ✨ :sparkles: for new features
# 📖 :book: for documentation
# ⚡ :zap: for performance improvements
# 🎨 :art: for code style/formatting
# 🧪 :test_tube: for tests
# 🔧 :wrench: for configuration changes
```

### 4. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request from your branch to `main`
3. Fill out the PR template completely
4. Wait for review and address feedback

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def detect_objects(image: np.ndarray, model: torch.nn.Module) -> List[Detection]:
    """Detect objects in an image using YOLOv7.
    
    Args:
        image: Input image as numpy array
        model: Trained YOLOv7 model
        
    Returns:
        List of detected objects
    """
    pass

# Use descriptive variable names
batch_size = 32  # Good
bs = 32  # Avoid

# GPU-specific considerations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add GPU-specific notes where relevant
- Use emojis sparingly but effectively
- Update README if adding new features

## Testing

### Writing Tests

```python
import pytest
import torch

def test_model_inference():
    """Test model inference on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = load_model('yolov7.pt')
    model.to('cuda')
    
    # Test code here
    assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run GPU-specific tests only
pytest -m gpu

# Run tests in parallel
pytest -n auto
```

## Documentation

### Adding Documentation

- Update README for new features
- Add docstrings to all functions
- Include usage examples
- Document GPU requirements and optimizations

### Building Docs Locally

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```

## Performance Considerations

When contributing, keep these GPU performance tips in mind:

1. **Memory Management**: Use `torch.cuda.empty_cache()` appropriately
2. **Batch Processing**: Optimize for larger batch sizes when possible
3. **Data Loading**: Use `pin_memory=True` for faster GPU transfers
4. **Mixed Precision**: Support AMP training where applicable
5. **Multi-GPU**: Ensure code works with DataParallel/DistributedDataParallel

## Recognition

Contributors will be recognized in:
- README acknowledgements
- Release notes
- Contributor list

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion for broader topics
- Reach out to maintainers directly

Thank you for contributing! 🚀

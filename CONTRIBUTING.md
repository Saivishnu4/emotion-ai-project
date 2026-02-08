# Contributing to Emotion AI

Thank you for your interest in contributing to Emotion AI! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, TensorFlow version)
- Error messages and stack traces

### Suggesting Enhancements

We welcome enhancement suggestions! Please create an issue with:
- Clear description of the enhancement
- Use cases and benefits
- Possible implementation approach

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/emotion-ai-project.git
   cd emotion-ai-project
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: descriptive commit message"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description
   - Reference related issues
   - Ensure all tests pass

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Comment complex logic

### Example:
```python
def preprocess_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image array
        target_size: Target dimensions (height, width)
        
    Returns:
        Preprocessed image array
    """
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    return image
```

## 🧪 Testing

- Write unit tests for new features
- Ensure existing tests pass
- Aim for >80% code coverage

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

- Update README.md for major features
- Add docstrings to all functions
- Update API documentation
- Include usage examples

## 🏗️ Project Structure

When adding new features, follow the existing structure:
```
src/
├── models/      # Model architectures
├── training/    # Training scripts
├── data/        # Data processing
└── utils/       # Utility functions
```

## ✅ Checklist Before Submitting PR

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts

## 📧 Questions?

Feel free to open an issue for any questions or reach out to:
- Email: saivishnusarode@gmail.com
- LinkedIn: [linkedin.com/in/saivishnu2002](https://www.linkedin.com/in/saivishnu2002)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Emotion AI! 🎉

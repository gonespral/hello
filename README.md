# Facial Recognition Project Facial Recognition

## Overview
Facial Recognition Project is a facial recognition system leveraging state-of-the-art libraries like PyTorch, OpenCV, and Scikit-learn. It allows real-time facial authentication and detection using pre-trained models.

## Features
- Real-time video processing with OpenCV
- Lightweight and optimized detection pipeline
- Scalable design for training custom facial datasets

## Environment Setup
Use **uv** for managing environments:

```bash
uv new <your-env-name>
uv install
```

## Usage
1. Train the model with your facial datasets:
   ```bash
   python main.py train
   ```
2. Run facial recognition:
   ```bash
   python main.py recognize
   ```

## Dependencies
All project dependencies are managed through the `pyproject.toml` file. Install dependencies from the `uv` environment.

## Contributing
1. Clone the repository:
   ```bash
git clone <repository_url>
   ```
2. Install dependencies with `uv`.
3. Submit PRs for enhancements or bug fixes.

## License
MIT License - See LICENSE file for details.
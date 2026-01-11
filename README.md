# Image Processing on Google Cloud Platform

## Project Overview
This project implements a parallel image processing pipeline using Python to apply five distinct filters (Grayscale, Gaussian Blur, Sobel Edge Detection, Sharpening, and Brightness) to the Food-101 dataset. The system compares two parallel paradigms: `multiprocessing` and `concurrent.futures`.

## Image Processing Pipeline
1. **Grayscale**: Luminance conversion via OpenCV.
2. **Gaussian Blur**: Noise reduction using a 3x3 kernel.
3. **Edge Detection**: Sobel filter to identify intensity gradients.
4. **Sharpening**: Custom 2D convolution kernel.
5. **Brightness**: Contrast and bias adjustment.

## GCP Deployment
- **Instance Name**: [instance-20251222-083623]
- **Instance ID**: [1205457453167459785]
- **Machine Type**: [e2-standard-4 (4 vCPUs, 16 GB RAM)]
- **Location**: [us-central1-c]
- **OS**: [Ubuntu 24.04 LTS (Noble Numbat)]
- **Architecture**: [X86_64]

## Performance Analysis
The following metrics were used to analyze scalability:
- **Speedup**: $T_1 / T_n$ (Ideal is $n$)
- **Efficiency**: $(Speedup / n) \times 100$

## How to Run
1. **Clone the Repository**: git clone <repo-link>
2. **Install Dependencies**: pip install opencv-python-headless numpy psutil
3. **Execute Benchmarks**: python3 image_processing.py
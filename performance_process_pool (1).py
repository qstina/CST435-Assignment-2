import cv2
import numpy as np
import concurrent.futures
import os
import time
import psutil

# Configuration matches your extracted Food-101 subset
INPUT_DIR = "images/food_subset"
OUTPUT_DIR = "processed_processes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_image_pipeline(img_path):
    """
    Worker function for image filtering.
    Applies the 5 required filters from Assignment 2 [cite: 187-192].
    """
    start_task = time.time()
    img = cv2.imread(img_path)
    if img is None: return None

    # 1. Grayscale Conversion 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. Gaussian Blur (3x3 kernel) 
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. Edge Detection (Sobel)
    edges = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=3)
    # 4. Image Sharpening 
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp = cv2.filter2D(gray, -1, kernel)
    # 5. Brightness Adjustment 
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)

    # Resource tracking: PID and Core ID for Analysis 
    pid = os.getpid()
    core_id = psutil.Process().cpu_num()
    
    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"proc_{filename}"), sharp)
    
    return {
        "pid": pid,
        "core": core_id,
        "duration": time.time() - start_task
    }

def run_test(image_list, worker_count):
    """Benchmarks execution time for a specific worker count[cite: 127]."""
    print(f"\n{'='*10} Testing with {worker_count} Processes {'='*10}")
    start_time = time.time()
    
    # Replacing ThreadPoolExecutor with ProcessPoolExecutor 
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(apply_image_pipeline, image_list))
            
    total_time = time.time() - start_time
    return total_time

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found.")
    else:
        all_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith('.jpg')]
        
        # Test counts: 1, 2, and 4,8 
        counts = [1, 2, 4, 8]
        benchmarks = {}

        for n in counts:
            duration = run_test(all_files, n)
            benchmarks[n] = duration
            print(f"Total Time for {n} process(es): {duration:.4f} seconds")

        # Performance Summary for Technical Report 
        print(f"\n{'='*20} Performance Analysis Summary (ProcessPool) {'='*20}")
        t1 = benchmarks[1] 
        for n, tn in benchmarks.items():
            speedup = t1 / tn
            efficiency = (speedup / n) * 100
            print(f"Processes: {n} | Time: {tn:.4f}s | Speedup: {speedup:.2f} | Efficiency: {efficiency:.2f}%")
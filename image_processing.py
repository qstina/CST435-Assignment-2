import os
import time
import multiprocessing as mp
import concurrent.futures
import cv2
import numpy as np

# --- Configuration ---
INPUT_DIR = os.path.expanduser("~/data")
OUTPUT_DIR = os.path.expanduser("~/processed_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_image_pipeline(args):
    """
    Worker function applying the 5 required manual filters.
    args: (input_path, output_path, paradigm_suffix)
    """
    in_p, out_p_base, suffix = args
    
    # Generate specific filename: original_name_suffix.jpg
    base_name = os.path.splitext(os.path.basename(in_p))[0]
    final_out_p = os.path.join(OUTPUT_DIR, f"{base_name}_{suffix}.jpg")

    try:
        img = cv2.imread(in_p)
        if img is None: return False

        # 1. Grayscale Conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur (3x3 kernel)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 3. Edge Detection (Sobel)
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.convertScaleAbs(edges)
        
        # 4. Image Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(gray, -1, kernel)
        
        # 5. Brightness Adjustment
        bright = cv2.convertScaleAbs(sharp, alpha=1.2, beta=30)
        
        cv2.imwrite(final_out_p, bright)
        return True
    except Exception:
        return False

def run_multiprocessing(tasks, cores):
    """Executes the pipeline using multiprocessing.Pool."""
    # Add 'multiprocess' suffix to each task
    mp_tasks = [(t[0], t[1], "multiprocess") for t in tasks]
    start = time.time()
    with mp.Pool(processes=cores) as pool:
        pool.map(apply_image_pipeline, mp_tasks)
    return time.time() - start

def run_concurrent_futures(tasks, cores):
    """Executes the pipeline using concurrent.futures.ProcessPoolExecutor."""
    # Add 'concurrent' suffix to each task
    cf_tasks = [(t[0], t[1], "concurrent") for t in tasks]
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        list(executor.map(apply_image_pipeline, cf_tasks))
    return time.time() - start

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found.")
        exit()

    tasks = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg')):
                tasks.append((os.path.join(root, f), os.path.join(OUTPUT_DIR, f)))

    print(f"Dataset Size: {len(tasks)} images. Beginning Performance Analysis...")

    cpu_limit = mp.cpu_count()
    test_configs = sorted(list(set([1, 2, 4, cpu_limit])))
    benchmarks = {"MP": {}, "CF": {}}

    # Benchmark Loop
    for c in test_configs:
        benchmarks["MP"][c] = run_multiprocessing(tasks, c)
        benchmarks["CF"][c] = run_concurrent_futures(tasks, c)

    # --- Performance Summary Table ---
    print("\n" + "="*100)
    print(f"{'Cores':<6} | {'Paradigm':<15} | {'Time (s)':<10} | {'Speedup':<10} | {'Efficiency (%)':<15}")
    print("-" * 100)
    
    for key, name in [("MP", "Multiprocessing"), ("CF", "Conc. Futures")]:
        t1 = benchmarks[key][1]
        for c in test_configs:
            tn = benchmarks[key][c]
            speedup = t1 / tn if tn > 0 else 0
            efficiency = (speedup / c) * 100 if c > 0 else 0
            print(f"{c:<6} | {name:<15} | {tn:<10.2f} | {speedup:<10.2f} | {efficiency:<15.2f}")
    print("="*100)
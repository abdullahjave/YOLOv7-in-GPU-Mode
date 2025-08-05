#!/usr/bin/env python3
"""
YOLOv7 GPU Mode Demo Script
==========================

A comprehensive demo script showcasing YOLOv7 GPU capabilities.
This script demonstrates:
- GPU detection and setup
- Model loading and inference
- Image and video processing
- Performance benchmarking
- Results visualization

Usage:
    python demo.py --source image.jpg          # Single image
    python demo.py --source video.mp4          # Video file
    python demo.py --source 0                  # Webcam
    python demo.py --benchmark                 # Performance test
"""

import argparse
import time
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_coords
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device
    from utils.datasets import letterbox
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔍 Make sure you're running from the YOLOv7 directory")
    sys.exit(1)

class YOLOv7Demo:
    """YOLOv7 Demo class with GPU acceleration."""
    
    def __init__(self, weights='weights/yolov7.pt', device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        """Initialize YOLOv7 model."""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        
        # Setup device
        self.device = select_device(device)
        print(f"🎮 Using device: {self.device}")
        
        # Load model
        print(f"📦 Loading model from {weights}...")
        self.model = attempt_load(weights, map_location=self.device)
        
        # Get model info
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Warmup
        print("🔥 Warming up model...")
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(img)
        print("✅ Model ready!")
    
    def preprocess(self, img0):
        """Preprocess image for inference."""
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    
    def detect(self, img0):
        """Run detection on image."""
        # Preprocess
        img = self.preprocess(img0)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # Process detections
        detections = []
        for det in pred:
            if len(det):
                # Rescale boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls in det:
                    detections.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class': int(cls),
                        'name': self.names[int(cls)]
                    })
        
        return detections
    
    def draw_results(self, img, detections):
        """Draw detection results on image."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            name = det['name']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return img
    
    def process_image(self, source, save_path=None):
        """Process single image."""
        print(f"📸 Processing image: {source}")
        
        # Load image
        img0 = cv2.imread(str(source))
        if img0 is None:
            print(f"❌ Could not load image: {source}")
            return
        
        # Run detection
        start_time = time.time()
        detections = self.detect(img0)
        inference_time = time.time() - start_time
        
        # Draw results
        img_result = self.draw_results(img0.copy(), detections)
        
        # Print results
        print(f"⚡ Inference time: {inference_time*1000:.1f}ms")
        print(f"🎯 Found {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['name']}: {det['confidence']:.2f}")
        
        # Save or display
        if save_path:
            cv2.imwrite(str(save_path), img_result)
            print(f"💾 Saved result to: {save_path}")
        else:
            cv2.imshow('YOLOv7 Detection', img_result)
            print("👀 Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_video(self, source, save_path=None):
        """Process video file or webcam."""
        print(f"🎬 Processing video: {source}")
        
        # Open video
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
            print("📷 Using webcam")
        else:
            cap = cv2.VideoCapture(str(source))
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                start_time = time.time()
                detections = self.detect(frame)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Draw results
                frame_result = self.draw_results(frame.copy(), detections)
                
                # Add FPS info
                fps_text = f"FPS: {1/inference_time:.1f}"
                cv2.putText(frame_result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save or display
                if writer:
                    writer.write(frame_result)
                else:
                    cv2.imshow('YOLOv7 Video Detection', frame_result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_count % 30 == 0:
                    avg_fps = frame_count / total_time
                    print(f"📊 Frame {frame_count}/{total_frames}, Avg FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"💾 Saved video to: {save_path}")
            cv2.destroyAllWindows()
            
            # Final stats
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"🏁 Processed {frame_count} frames")
            print(f"📊 Average FPS: {avg_fps:.1f}")
    
    def benchmark(self, iterations=100):
        """Run performance benchmark."""
        print(f"🏎️ Running benchmark ({iterations} iterations)...")
        
        # Create test image
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            _ = self.detect(test_img)
        
        # Benchmark
        times = []
        for i in range(iterations):
            start_time = time.time()
            _ = self.detect(test_img)
            torch.cuda.synchronize()  # Wait for GPU
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"📊 Progress: {i+1}/{iterations}")
        
        # Results
        times = np.array(times)
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        avg_fps = 1000 / avg_time
        
        print("\n🏆 Benchmark Results:")
        print(f"📊 Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"⚡ Average FPS: {avg_fps:.1f}")
        print(f"🚀 Min time: {min_time:.2f} ms ({1000/min_time:.1f} FPS)")
        print(f"🐌 Max time: {max_time:.2f} ms ({1000/max_time:.1f} FPS)")
        
        return {
            'avg_time_ms': avg_time,
            'avg_fps': avg_fps,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time
        }

def check_gpu():
    """Check GPU availability and info."""
    print("🔍 GPU Information:")
    print("=" * 40)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"🎮 CUDA Version: {torch.version.cuda}")
        print(f"🔢 GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"🏷  GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
        
        # Memory info
        if torch.cuda.device_count() > 0:
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached, {memory_total:.1f}GB total")
    else:
        print("❌ CUDA not available")
        print("🔍 Make sure NVIDIA drivers and CUDA are installed")
    
    print()

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='YOLOv7 GPU Mode Demo')
    parser.add_argument('--weights', type=str, default='weights/yolov7.pt', help='model weights path')
    parser.add_argument('--source', type=str, help='source image/video path or webcam index')
    parser.add_argument('--output', type=str, help='output path for saving results')
    parser.add_argument('--img-size', type=int, default=640, help='inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--benchmark', action='store_true', help='run performance benchmark')
    parser.add_argument('--no-gpu-check', action='store_true', help='skip GPU information display')
    
    args = parser.parse_args()
    
    # Welcome message
    print("🚀 YOLOv7 GPU Mode Demo")
    print("=" * 40)
    
    # Check GPU
    if not args.no_gpu_check:
        check_gpu()
    
    # Initialize model
    try:
        demo = YOLOv7Demo(
            weights=args.weights,
            device=args.device,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        print("🔍 Make sure the weights file exists and is valid")
        return
    
    # Run benchmark
    if args.benchmark:
        demo.benchmark()
        return
    
    # Process source
    if args.source is None:
        print("❌ No source specified. Use --source or --benchmark")
        print("Examples:")
        print("  python demo.py --source image.jpg")
        print("  python demo.py --source video.mp4")
        print("  python demo.py --source 0  # webcam")
        print("  python demo.py --benchmark")
        return
    
    # Determine if source is image or video
    source_path = Path(args.source)
    if args.source.isdigit() or source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video processing
        demo.process_video(args.source, args.output)
    else:
        # Image processing
        demo.process_image(args.source, args.output)
    
    print("🎉 Demo completed successfully!")

if __name__ == '__main__':
    main()

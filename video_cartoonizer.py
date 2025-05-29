import cv2
import numpy as np
import os
import sys
from pathlib import Path

class VideoCartoonizer:
    def __init__(self):
        self.bilateral_filters = 7
        self.blur_value = 7
        self.line_size = 7
        
    def edge_mask(self, img, line_size, blur_value):
        """Create edge mask for cartoon effect"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, line_size, blur_value)
        return edges

    def color_quantization(self, img, k=8):
        """Reduce colors for cartoon effect"""
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        return res.reshape(img.shape)

    def cartoonize_frame(self, img):
        """Apply cartoon effect to single frame"""
        # Bilateral filter to reduce noise while keeping edges sharp
        for _ in range(self.bilateral_filters):
            img = cv2.bilateralFilter(img, 9, 80, 80)
        
        # Create edge mask
        edges = self.edge_mask(img, self.line_size, self.blur_value)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Color quantization
        img = self.color_quantization(img)
        
        # Combine quantized image with edges
        cartoon = cv2.bitwise_and(img, edges)
        
        return cartoon

    def process_video(self, input_path, output_path=None):
        """Process entire video"""
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found!")
            return False
            
        if output_path is None:
            name, ext = os.path.splitext(input_path)
            output_path = f"{name}_cartoon{ext}"
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video '{input_path}'")
            return False
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved as: {output_path}")
        print(f"Total frames: {total_frames}")
        print("Processing...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply cartoon effect
            cartoon_frame = self.cartoonize_frame(frame)
            
            # Write frame
            out.write(cartoon_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing complete! Output saved as: {output_path}")
        return True

def main():
    print("=== Video Cartoonizer ===")
    print("This tool converts your video to cartoon style")
    print()
    
    cartoonizer = VideoCartoonizer()
    
    if len(sys.argv) > 1:
        # Command line argument provided
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            output_file = None
    else:
        # Interactive mode
        input_file = input("Enter path to your video file: ").strip()
        output_file = input("Enter output filename (press Enter for auto-naming): ").strip()
        if not output_file:
            output_file = None
    
    # Process the video
    success = cartoonizer.process_video(input_file, output_file)
    
    if success:
        print("\n✅ Success! Your cartoon video is ready!")
    else:
        print("\n❌ Failed to process video. Please check the file path and format.")

if __name__ == "__main__":
    main()
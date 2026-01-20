#!/usr/bin/env python3
"""
Simple chessboard calibration using uncompressed YUYV format.
Designed for Raspberry Pi 5 with stereo camera setup.
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
import argparse


def setup_camera_yuyv(camera_index=0, width=640, height=480):
    """
    Setup camera with YUYV uncompressed format.
    
    Args:
        camera_index: Camera index (0 or 1 for stereo setup)
        width: Image width
        height: Image height
    
    Returns:
        Configured Picamera2 object
    """
    picam2 = Picamera2(camera_index)
    
    # Configure camera with YUYV format (uncompressed)
    config = picam2.create_still_configuration(
        main={
            "size": (width, height),
            "format": "YUYV"  # Uncompressed YUYV format instead of MJPEG
        }
    )
    picam2.configure(config)
    picam2.start()
    
    # Allow camera to warm up
    time.sleep(2)
    
    return picam2


def capture_calibration_images(picam2, chessboard_size, num_images=20):
    """
    Capture calibration images with chessboard detection.
    
    Args:
        picam2: Picamera2 object
        chessboard_size: Tuple of (rows, cols) for chessboard internal corners
        num_images: Number of calibration images to capture
    
    Returns:
        List of object points and image points for calibration
    """
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    images_captured = 0
    
    print(f"Starting calibration capture. Need {num_images} valid images.")
    print("Press 'c' to capture, 'q' to quit early")
    
    while images_captured < num_images:
        # Capture frame from YUYV format
        frame = picam2.capture_array()
        
        # Convert YUYV to BGR for OpenCV processing
        if frame.shape[2] == 2:  # YUYV format has 2 channels
            gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_YUYV)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points and image points
        if ret:
            # Draw corners for visualization
            cv2.drawChessboardCorners(display_frame, chessboard_size, corners, ret)
            cv2.putText(display_frame, f"Chessboard detected! Press 'c' to capture ({images_captured}/{num_images})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"No chessboard detected ({images_captured}/{num_images})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            images_captured += 1
            print(f"Captured image {images_captured}/{num_images}")
        elif key == ord('q'):
            print("Calibration cancelled by user")
            break
    
    cv2.destroyAllWindows()
    return objpoints, imgpoints, gray.shape[::-1]


def calibrate_camera(objpoints, imgpoints, image_size):
    """
    Perform camera calibration.
    
    Args:
        objpoints: List of object points
        imgpoints: List of image points
        image_size: Size of the image (width, height)
    
    Returns:
        Camera matrix, distortion coefficients, and reprojection error
    """
    if len(objpoints) == 0:
        print("No valid calibration images captured!")
        return None, None, None
    
    print(f"\nPerforming calibration with {len(objpoints)} images...")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    return mtx, dist, mean_error


def save_calibration(camera_matrix, dist_coeffs, filename="calibration.npz"):
    """
    Save calibration parameters to file.
    
    Args:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        filename: Output filename
    """
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\nCalibration saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Chessboard calibration using YUYV format")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Image height (default: 480)")
    parser.add_argument("--rows", type=int, default=6, help="Chessboard rows (internal corners, default: 6)")
    parser.add_argument("--cols", type=int, default=9, help="Chessboard columns (internal corners, default: 9)")
    parser.add_argument("--images", type=int, default=20, help="Number of calibration images (default: 20)")
    parser.add_argument("--output", type=str, default="calibration.npz", help="Output filename (default: calibration.npz)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chessboard Calibration - YUYV Format")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Chessboard: {args.rows}x{args.cols} internal corners")
    print(f"Format: YUYV (uncompressed)")
    print("=" * 60)
    
    try:
        # Setup camera with YUYV format
        picam2 = setup_camera_yuyv(args.camera, args.width, args.height)
        
        # Capture calibration images
        objpoints, imgpoints, image_size = capture_calibration_images(
            picam2, (args.rows, args.cols), args.images
        )
        
        # Stop camera
        picam2.stop()
        
        # Perform calibration
        mtx, dist, error = calibrate_camera(objpoints, imgpoints, image_size)
        
        if mtx is not None:
            print("\n" + "=" * 60)
            print("Calibration Results:")
            print("=" * 60)
            print(f"Reprojection Error: {error:.4f} pixels")
            print("\nCamera Matrix:")
            print(mtx)
            print("\nDistortion Coefficients:")
            print(dist)
            
            # Save calibration
            save_calibration(mtx, dist, args.output)
            print("\nCalibration completed successfully!")
        else:
            print("\nCalibration failed - insufficient valid images")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

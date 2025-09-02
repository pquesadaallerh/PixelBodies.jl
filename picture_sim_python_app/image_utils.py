import os
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence
import pygame

def load_cached_bbox(cache_file: Path):
    """Load cached bounding box if it exists."""
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    return None

def save_bbox_cache(cache_file: Path, bbox_data: dict):
    """Save bounding box to cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(bbox_data, f, indent=2)

def capture_image(
        input_folder: str | Path,
        image_name: str = "input.png",
        fixed_aspect_ratio: tuple = None,
        fixed_size: tuple = None,
        selection_box_mode: bool = True,
        saved_selection: tuple = None,
        use_cached_box: bool = False,
        **_kwargs, # To allow camera_index arg even if not used (only needed for windows version of the function,
                  # see image_utils_windows.py)
) -> tuple:
    """
    Capture image from webcam with optional fixed aspect ratio, size, or interactive selection box.
    Now includes automatic caching of bounding box selections.
    
    Args:
        input_folder: Folder to save the captured image
        image_name: Name of the output image file
        fixed_aspect_ratio: If provided (width, height), enforces this aspect ratio
        fixed_size: If provided (width, height), captures image at this exact size
        selection_box_mode: If True, shows a click-and-drag selection with fixed aspect ratio
        saved_selection: If provided (x, y, w, h), uses this as the initial selection box
        use_cached_box: If True, automatically uses cached bounding box without user interaction
        
    Returns:
        tuple: (x, y, w, h) coordinates of the selected region, or None if cancelled
    """
    # Setup cache file path
    cache_dir = Path(input_folder).parent / "cache"
    cache_file = cache_dir / "bbox_cache.json"
    
    # If use_cached_box is True, try to load and use cached box automatically
    if use_cached_box and selection_box_mode:
        cached_data = load_cached_bbox(cache_file)
        if cached_data:
            saved_selection = tuple(cached_data.get("bbox", (0, 0, 0, 0)))
            print(f"Using cached bounding box: {saved_selection}")
            
            # Capture image automatically using cached bbox
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Webcam not found or cannot be opened.")
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError("Failed to capture frame from webcam.")
            
            # Apply cached bounding box
            box_x, box_y, box_w, box_h = saved_selection
            
            # Ensure bounding box is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            box_x = max(0, min(frame_w - 1, box_x))
            box_y = max(0, min(frame_h - 1, box_y))
            box_w = max(1, min(frame_w - box_x, box_w))
            box_h = max(1, min(frame_h - box_y, box_h))
            
            # Crop and save
            selected_region = frame[box_y:box_y + box_h, box_x:box_x + box_w]
            path = Path(input_folder) / image_name
            cv2.imwrite(str(path), selected_region)
            print(f"Image automatically captured using cached bbox and saved to {path}")
            
            return (box_x, box_y, box_w, box_h)
        else:
            print("No cached bounding box found. Falling back to interactive mode.")
            # Fall through to interactive mode
    
    # Try to load cached bounding box if no saved_selection provided (interactive mode)
    if saved_selection is None and selection_box_mode and not use_cached_box:
        cached_data = load_cached_bbox(cache_file)
        if cached_data:
            saved_selection = tuple(cached_data.get("bbox", (0, 0, 0, 0)))
            print(f"Found cached bounding box: {saved_selection}")
            print("Press ENTER to use cached selection, or 'r' to reselect:")
            
            # Simple input check - if user wants to reselect
            try:
                user_input = input().strip().lower()
                if user_input == 'r':
                    saved_selection = None
                    print("Will create new selection...")
                else:
                    print("Using cached selection...")
            except KeyboardInterrupt:
                print("Using cached selection...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found or cannot be opened.")

    if selection_box_mode and fixed_aspect_ratio:
        print("ASPECT RATIO SELECTION MODE:")
        if saved_selection:
            print("- Previous selection loaded as default")
            print("- Press [space] to use current selection")
        print("- Click and drag to define a new selection area (constant aspect ratio)")
        print("- After selection, use [w/a/s/d] to fine-tune position")
        print("- Press [r] to reset and start over")
        print("- Press [space] to capture the selected region")
        print("- Press [ESC] to quit")
        print(f"- Target aspect ratio: {fixed_aspect_ratio[0]}:{fixed_aspect_ratio[1]}")
        
        # Variables for click-and-drag selection
        target_ratio = fixed_aspect_ratio[0] / fixed_aspect_ratio[1]
        drawing = False
        selection_made = False
        start_point = None
        
        # Initialize with saved selection if provided
        if saved_selection:
            box_x, box_y, box_w, box_h = saved_selection
            selection_made = True
            print(f"Using previous selection: {box_w}x{box_h} at ({box_x}, {box_y})")
        else:
            box_x, box_y, box_w, box_h = 0, 0, 0, 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, box_x, box_y, box_w, box_h, selection_made
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                selection_made = False
                start_point = (x, y)
                box_x, box_y, box_w, box_h = x, y, 0, 0
                
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                if start_point:
                    # Calculate the width and height from drag
                    drag_w = abs(x - start_point[0])
                    drag_h = abs(y - start_point[1])
                    
                    # Prevent division by zero and handle very small movements
                    if drag_w < 10 and drag_h < 10:
                        # Too small to determine aspect ratio, skip update
                        return
                    elif drag_h == 0:
                        # Purely horizontal drag - use width to determine height
                        box_w = drag_w
                        box_h = int(box_w / target_ratio)
                    elif drag_w == 0:
                        # Purely vertical drag - use height to determine width
                        box_h = drag_h
                        box_w = int(box_h * target_ratio)
                    else:
                        # Normal case - determine which dimension constrains the box
                        if drag_w / drag_h > target_ratio:
                            # Width is the limiting factor
                            box_h = drag_h
                            box_w = int(box_h * target_ratio)
                        else:
                            # Height is the limiting factor  
                            box_w = drag_w
                            box_h = int(box_w / target_ratio)
                    
                    # Position the box based on drag direction
                    if x >= start_point[0]:  # Dragging right
                        box_x = start_point[0]
                    else:  # Dragging left
                        box_x = start_point[0] - box_w
                        
                    if y >= start_point[1]:  # Dragging down
                        box_y = start_point[1]
                    else:  # Dragging up
                        box_y = start_point[1] - box_h
                    
                    # Keep box within camera bounds
                    if box_x < 0:
                        box_x = 0
                    if box_y < 0:
                        box_y = 0
                    if box_x + box_w > cap.get(cv2.CAP_PROP_FRAME_WIDTH):
                        box_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) - box_w)
                    if box_y + box_h > cap.get(cv2.CAP_PROP_FRAME_HEIGHT):
                        box_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - box_h)
                        
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if box_w > 0 and box_h > 0:
                    selection_made = True
                    print("Selection made! Use [w/a/s/d] to adjust position, [r] to reset, [space] to capture.")
        
        cv2.namedWindow("Live Feed")
        cv2.setMouseCallback("Live Feed", mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            display_frame = frame.copy()
            
            # Draw the selection box if we have valid dimensions
            if box_w > 0 and box_h > 0:
                # Change color based on selection state
                color = (0, 255, 0) if selection_made else (255, 255, 0)  # Green if selected, yellow while dragging
                cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
                
                # Draw corner markers
                corner_size = 10
                corners = [
                    (box_x, box_y), (box_x + box_w, box_y),
                    (box_x, box_y + box_h), (box_x + box_w, box_y + box_h)
                ]
                for (cx, cy) in corners:
                    cv2.line(display_frame, (cx - corner_size, cy), (cx + corner_size, cy), color, 2)
                    cv2.line(display_frame, (cx, cy - corner_size), (cx, cy + corner_size), color, 2)
                
                # Add text overlay with status
                status = "SELECTED - Use WASD to adjust" if selection_made else "DRAGGING"
                cv2.putText(display_frame, f"Box: {box_w}x{box_h} - {status}", (box_x, box_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow("Live Feed", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Aborted.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 32:  # Spacebar
                if box_w > 0 and box_h > 0:
                    # Capture the selected region
                    selected_region = frame[box_y:box_y + box_h, box_x:box_x + box_w]
                    path = input_folder / image_name
                    cv2.imwrite(str(path), selected_region)
                    print(f"Image saved to {path}")
                    
                    # Save bounding box to cache
                    bbox_data = {
                        'bbox': (box_x, box_y, box_w, box_h),
                        'aspect_ratio': fixed_aspect_ratio,
                        'image_name': image_name
                    }
                    save_bbox_cache(cache_file, bbox_data)
                    print(f"Bounding box cached for future use")
                    
                    break
                else:
                    print("Please select a region first by clicking and dragging.")
            
            # WASD controls for fine-tuning position (only after selection is made)
            if selection_made:
                move_step = 5  # Smaller step for fine adjustment
                cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if key == ord('w'):  # Move up
                    box_y = max(0, box_y - move_step)
                elif key == ord('s'):  # Move down
                    box_y = min(cam_h - box_h, box_y + move_step)
                elif key == ord('a'):  # Move left
                    box_x = max(0, box_x - move_step)
                elif key == ord('d'):  # Move right
                    box_x = min(cam_w - box_w, box_x + move_step)
                elif key == ord('r'):  # Reset selection
                    selection_made = False
                    box_x, box_y, box_w, box_h = 0, 0, 0, 0
                    print("Selection reset. Click and drag to make a new selection.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Return the coordinates of the final selection
        return (box_x, box_y, box_w, box_h)
        
    else:
        # Original behavior for non-selection modes
        print("Press [space] to capture image, or [ESC] to quit.")
        if fixed_aspect_ratio:
            print(f"Fixed aspect ratio mode: {fixed_aspect_ratio[0]}:{fixed_aspect_ratio[1]}")
        if fixed_size:
            print(f"Fixed size mode: {fixed_size[0]}x{fixed_size[1]} pixels")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Aborted.")
                cap.release()
                cv2.destroyAllWindows()
                return (0, 0, 0, 0)  # Return empty coordinates when aborted
            elif key == 32:  # Spacebar
                path = input_folder / image_name
                cv2.imwrite(str(path), frame)
                print(f"Image saved to {path}")
                break

        cap.release()
        cv2.destroyAllWindows()

        # Apply post-processing for non-selection modes
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load image {path} for cropping.")

        if fixed_size:
            # Crop to exact fixed size from center
            h, w = img.shape[:2]
            target_w, target_h = fixed_size
            
            start_x = max(0, (w - target_w) // 2)
            start_y = max(0, (h - target_h) // 2)
            end_x = min(w, start_x + target_w)
            end_y = min(h, start_y + target_h)
            
            cropped_img = img[start_y:end_y, start_x:end_x]
            
            if cropped_img.shape[:2] != (target_h, target_w):
                padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                paste_y = (target_h - cropped_img.shape[0]) // 2
                paste_x = (target_w - cropped_img.shape[1]) // 2
                padded_img[paste_y:paste_y+cropped_img.shape[0], paste_x:paste_x+cropped_img.shape[1]] = cropped_img
                cropped_img = padded_img
                
            cv2.imwrite(str(path), cropped_img)
            print(f"Fixed size image ({target_w}x{target_h}) saved to {path}")
            
            # Return the crop coordinates for fixed size mode
            return (start_x, start_y, target_w, target_h)
            
        elif fixed_aspect_ratio:
            # Crop to fixed aspect ratio
            h, w = img.shape[:2]
            target_ratio = fixed_aspect_ratio[0] / fixed_aspect_ratio[1]
            current_ratio = w / h
            
            if current_ratio > target_ratio:
                new_width = int(h * target_ratio)
                start_x = (w - new_width) // 2
                cropped_img = img[:, start_x:start_x + new_width]
            else:
                new_height = int(w / target_ratio)
                start_y = (h - new_height) // 2
                cropped_img = img[start_y:start_y + new_height, :]
            
            cv2.imwrite(str(path), cropped_img)
            print(f"Fixed aspect ratio image ({fixed_aspect_ratio[0]}:{fixed_aspect_ratio[1]}) saved to {path}")
            
            # Return the crop coordinates for fixed aspect ratio mode
            h, w = cropped_img.shape[:2]
            return (0, 0, w, h)  # Since we processed the full image
        
        else:
            # Original manual cropping behavior
            print("Select ROI (drag mouse to crop). Press ENTER or SPACE to confirm, or 'c' to cancel.")
            roi = cv2.selectROI("Crop Image", img, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()

            x, y, w, h = roi
            if w > 0 and h > 0:
                cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
                cv2.imwrite(str(path), cropped_img)
                print(f"Cropped image saved to {path}")
                return (int(x), int(y), int(w), int(h))  # Return the manual crop coordinates
            else:
                print("No crop selected, original image kept.")
                h, w = img.shape[:2]
                return (0, 0, w, h)  # Return full image coordinates



def resize_gif(
        input_path: str | Path,
        output_path: str | Path,
        target_size: tuple=(1920, 1080),
        maintain_aspect: bool = True,
        fill_color: tuple=(0, 0, 0, 0),
) -> None:
    """
    Resize each frame of a GIF with improved aspect ratio handling.

    Args:
        input_path (str): Path to the input GIF.
        output_path (str): Path to save the resized GIF.
        target_size (tuple): Desired (width, height), e.g., (1920, 1080).
        maintain_aspect (bool): If True, maintain aspect ratio (may result in smaller output).
                               If False, stretch to exact target size.
        fill_color (tuple): RGBA fill color for padding when maintain_aspect=True.
    """
    with Image.open(input_path) as img:
        frames = []
        durations = []

        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGBA")
            orig_w, orig_h = frame.size
            target_w, target_h = target_size

            if maintain_aspect:
                # Compute scale to preserve aspect ratio
                scale = min(target_w / orig_w, target_h / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                # Resize with aspect ratio
                resized = frame.resize((new_w, new_h), Image.BICUBIC)

                # Create new canvas and paste resized frame in center
                canvas = Image.new("RGBA", target_size, fill_color)
                offset_x = (target_w - new_w) // 2
                offset_y = (target_h - new_h) // 2
                canvas.paste(resized, (offset_x, offset_y))
                frames.append(canvas)
            else:
                # Stretch to exact target size
                resized = frame.resize(target_size, Image.BICUBIC)
                frames.append(resized)

            durations.append(frame.info.get('duration', 100))

        # Save as animated GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2
        )


def get_gif_dimensions(gif_path: str | Path) -> tuple:
    """
    Get the dimensions of the first frame of a GIF.
    
    Args:
        gif_path: Path to the GIF file
        
    Returns:
        tuple: (width, height) of the GIF
    """
    with Image.open(gif_path) as img:
        return img.size


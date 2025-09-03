import os
import json
import subprocess
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import psutil
from PIL import Image, ImageSequence


def enumerate_cameras(max_cameras: int = 5, timeout_per_camera: float = 2.0) -> list:
    """
    Enumerate available cameras with timeout to prevent hanging.

    Args:
        max_cameras: Maximum number of camera indices to check
        timeout_per_camera: Timeout in seconds for each camera check

    Returns:
        list: List of dictionaries with camera info
    """
    available_cameras = []

    def test_camera(index, result_list):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set lower resolution for faster enumeration
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if cap.isOpened():
                # Quick test read
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    backend_name = cap.getBackendName()

                    result_list.append({
                        'index': index,
                        'name': f"Camera {index} ({backend_name})",
                        'resolution': (width, height)
                    })
            cap.release()
        except:
            pass  # Ignore any errors during enumeration

    # Test cameras with threading and timeout
    for i in range(max_cameras):
        result = []
        thread = threading.Thread(target=test_camera, args=(i, result))
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_per_camera)

        if thread.is_alive():
            # Thread timed out, skip this camera
            continue

        available_cameras.extend(result)

        # If we found cameras but this one failed, and we're past index 1, stop checking
        if not result and available_cameras and i > 1:
            break

    return available_cameras


def select_camera(auto_select_external: bool = True, silent: bool = False) -> int:
    """
    Select camera with optimized enumeration and fallback options.

    Args:
        auto_select_external: If True, automatically selects the highest index camera
        silent: If True, suppress all output

    Returns:
        int: Selected camera index
    """
    if not silent:
        print("Enumerating cameras...")

    cameras = enumerate_cameras()

    if not cameras:
        # Fallback: try common camera indices directly
        for i in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        if not silent:
                            print(f"Found working camera at index {i}")
                        return i
            except:
                continue
        raise RuntimeError("No working cameras found.")

    if len(cameras) == 1:
        selected = cameras[0]['index']
        if not silent:
            print(f"Using camera {selected}: {cameras[0]['name']}")
        return selected

    if auto_select_external:
        # Select the highest index camera (usually external)
        selected = cameras[-1]['index']
        if not silent:
            print(f"Auto-selected external camera {selected}: {cameras[-1]['name']}")
        return selected

    # Interactive selection
    if not silent:
        print("Available cameras:")
        for cam in cameras:
            print(f"  {cam['index']}: {cam['name']} - {cam['resolution'][0]}x{cam['resolution'][1]}")

    while True:
        try:
            choice = int(input(f"Select camera index: "))
            if any(cam['index'] == choice for cam in cameras):
                return choice
            print(f"Invalid choice. Available indices: {[cam['index'] for cam in cameras]}")
        except (ValueError, KeyboardInterrupt):
            # Default to first available camera
            return cameras[0]['index']


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
        camera_index: int = 1,
        auto_select_external: bool = True,
) -> tuple:
    """
    Capture image from webcam with optimized camera handling.
    """
    # Setup cache file path
    cache_dir = Path(input_folder).parent / "cache"
    cache_file = cache_dir / "bbox_cache.json"

    # Select camera if not specified
    if camera_index is None:
        try:
            camera_index = select_camera(auto_select_external, silent=use_cached_box)
        except RuntimeError as e:
            print(f"Camera selection failed: {e}")
            # Try default camera as last resort
            camera_index = 0

    # If use_cached_box is True, try to load and use cached box automatically
    if use_cached_box and selection_box_mode:
        cached_data = load_cached_bbox(cache_file)
        if cached_data:
            saved_selection = tuple(cached_data.get("bbox", (0, 0, 0, 0)))

            # Capture image automatically using cached bbox
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError(f"Camera {camera_index} not accessible.")

            # Optimize camera settings for performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Wait a moment for camera to initialize
            time.sleep(0.5)

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                raise RuntimeError("Failed to capture frame from camera.")

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

            return (box_x, box_y, box_w, box_h)

    # Try to load cached bounding box if no saved_selection provided (interactive mode)
    if saved_selection is None and selection_box_mode and not use_cached_box:
        cached_data = load_cached_bbox(cache_file)
        if cached_data:
            saved_selection = tuple(cached_data.get("bbox", (0, 0, 0, 0)))
            print("Press ENTER to use cached selection, or 'r' to reselect (5 sec timeout):")

            # Timeout for user input to prevent hanging
            def get_user_input():
                try:
                    return input().strip().lower()
                except:
                    return ""

            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError()

            try:
                # Set timeout for Windows (using threading as signal doesn't work well on Windows)
                result = []

                def input_thread():
                    try:
                        result.append(input().strip().lower())
                    except:
                        result.append("")

                thread = threading.Thread(target=input_thread)
                thread.daemon = True
                thread.start()
                thread.join(timeout=5.0)

                user_input = result[0] if result else ""
                if user_input == 'r':
                    saved_selection = None
            except:
                pass  # Use cached selection on any error

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} not accessible.")

    # Optimize camera settings for performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Wait for camera to stabilize
    time.sleep(0.5)

    if selection_box_mode and fixed_aspect_ratio:
        print("ASPECT RATIO SELECTION MODE:")
        if saved_selection:
            print("- Previous selection loaded")
        print("- Click+drag to select, [w/a/s/d] to adjust, [space] to capture, [ESC] to quit")

        # Variables for click-and-drag selection
        target_ratio = fixed_aspect_ratio[0] / fixed_aspect_ratio[1]
        drawing = False
        selection_made = False
        start_point = None

        # Initialize with saved selection if provided
        if saved_selection:
            box_x, box_y, box_w, box_h = saved_selection
            selection_made = True
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
                    drag_w = abs(x - start_point[0])
                    drag_h = abs(y - start_point[1])

                    if drag_w < 10 and drag_h < 10:
                        return
                    elif drag_h == 0:
                        box_w = drag_w
                        box_h = int(box_w / target_ratio)
                    elif drag_w == 0:
                        box_h = drag_h
                        box_w = int(box_h * target_ratio)
                    else:
                        if drag_w / drag_h > target_ratio:
                            box_h = drag_h
                            box_w = int(box_h * target_ratio)
                        else:
                            box_w = drag_w
                            box_h = int(box_w / target_ratio)

                    # Position the box based on drag direction
                    box_x = start_point[0] if x >= start_point[0] else start_point[0] - box_w
                    box_y = start_point[1] if y >= start_point[1] else start_point[1] - box_h

                    # Keep box within camera bounds
                    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    box_x = max(0, min(cam_w - box_w, box_x))
                    box_y = max(0, min(cam_h - box_h, box_y))

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if box_w > 0 and box_h > 0:
                    selection_made = True

        cv2.namedWindow("Live Feed", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Live Feed", mouse_callback)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Skip frames for better performance
            frame_count += 1
            if frame_count % 2 != 0:  # Only process every other frame
                continue

            display_frame = frame.copy()

            # Draw the selection box if we have valid dimensions
            if box_w > 0 and box_h > 0:
                color = (0, 255, 0) if selection_made else (255, 255, 0)
                cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)

                # Draw corner markers
                corners = [(box_x, box_y), (box_x + box_w, box_y), (box_x, box_y + box_h),
                           (box_x + box_w, box_y + box_h)]
                for (cx, cy) in corners:
                    cv2.line(display_frame, (cx - 10, cy), (cx + 10, cy), color, 2)
                    cv2.line(display_frame, (cx, cy - 10), (cx, cy + 10), color, 2)

                status = "SELECTED" if selection_made else "DRAGGING"
                cv2.putText(display_frame, f"{box_w}x{box_h} - {status}", (box_x, box_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Live Feed", display_frame)

            key = cv2.waitKey(30) & 0xFF  # Increase wait time for better performance
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 32:  # Spacebar
                if box_w > 0 and box_h > 0:
                    selected_region = frame[box_y:box_y + box_h, box_x:box_x + box_w]
                    path = Path(input_folder) / image_name
                    cv2.imwrite(str(path), selected_region)

                    # Save bounding box to cache
                    bbox_data = {
                        'bbox': (box_x, box_y, box_w, box_h),
                        'aspect_ratio': fixed_aspect_ratio,
                        'image_name': image_name,
                        'camera_index': camera_index
                    }
                    save_bbox_cache(cache_file, bbox_data)
                    break

            # WASD controls for fine-tuning position
            if selection_made:
                move_step = 5
                cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if key == ord('w'):
                    box_y = max(0, box_y - move_step)
                elif key == ord('s'):
                    box_y = min(cam_h - box_h, box_y + move_step)
                elif key == ord('a'):
                    box_x = max(0, box_x - move_step)
                elif key == ord('d'):
                    box_x = min(cam_w - box_w, box_x + move_step)
                elif key == ord('r'):
                    selection_made = False
                    box_x, box_y, box_w, box_h = 0, 0, 0, 0

        cap.release()
        cv2.destroyAllWindows()
        return (box_x, box_y, box_w, box_h)

    else:
        # Original behavior for non-selection modes
        print("Press [space] to capture, [ESC] to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Live Feed", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return (0, 0, 0, 0)
            elif key == 32:  # Spacebar
                path = Path(input_folder) / image_name
                cv2.imwrite(str(path), frame)
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
                padded_img[paste_y:paste_y + cropped_img.shape[0], paste_x:paste_x + cropped_img.shape[1]] = cropped_img
                cropped_img = padded_img

            cv2.imwrite(str(path), cropped_img)
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
            h, w = cropped_img.shape[:2]
            return (0, 0, w, h)

        else:
            # Original manual cropping behavior
            print("Select ROI (drag mouse to crop). Press ENTER or SPACE to confirm, or 'c' to cancel.")
            roi = cv2.selectROI("Crop Image", img, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()

            x, y, w, h = roi
            if w > 0 and h > 0:
                cropped_img = img[int(y):int(y + h), int(x):int(x + w)]
                cv2.imwrite(str(path), cropped_img)
                return (int(x), int(y), int(w), int(h))
            else:
                h, w = img.shape[:2]
                return (0, 0, w, h)


def resize_gif(
        input_path: str | Path,
        output_path: str | Path,
        target_size: tuple = (1920, 1080),
        maintain_aspect: bool = True,
        fill_color: tuple = (0, 0, 0, 0),
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


def find_display_processes():
    """Find running display processes by looking for our display scripts."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('persistent_gif_display' in str(arg) for arg in cmdline):
                processes.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def stop_display_processes():
    """Stop any running display processes."""
    pids = find_display_processes()
    stopped = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
            stopped.append(pid)
            print(f"[display] Stopped display process {pid}")
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                proc.kill()  # Force kill if terminate didn't work
                stopped.append(pid)
                print(f"[display] Force killed display process {pid}")
            except psutil.NoSuchProcess:
                pass
        except Exception as e:
            print(f"[display] Failed to stop process {pid}: {e}")
    return stopped

def start_display_process(script_dir: Path, monitor_index=1, use_qt_version=True, flip_90_degrees=False):
    """Start the display process."""
    try:
        if use_qt_version:
            if flip_90_degrees:
                script_path = script_dir / "persistent_gif_display" / "persistent_gif_display_90_deg.py"
            else:
                script_path = script_dir / "persistent_gif_display" / "persistent_gif_display_2.py"
        else:
            script_path = script_dir / "persistent_gif_display" / "persistent_gif_display.py"

        if not os.path.exists(script_path):
            raise FileNotFoundError

        # Start the process in the background
        proc = subprocess.Popen([
            "python", str(script_path), str(monitor_index)
        ], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)

        print(f"[display] Started display process {proc.pid} on monitor {monitor_index}")
        time.sleep(2)  # Give it time to start
        return proc.pid
    except Exception as e:
        print(f"[display] Failed to start display process: {e}")
        return None

def restart_display_process(script_dir: Path, monitor_index=1, use_qt_version=True, flip_90_degrees=False):
    """Stop existing display processes and start a new one."""
    print("[display] Restarting display process...")
    stopped_pids = stop_display_processes()
    time.sleep(1)  # Brief pause between stop and start
    new_pid = start_display_process(script_dir, monitor_index, use_qt_version, flip_90_degrees)
    return new_pid

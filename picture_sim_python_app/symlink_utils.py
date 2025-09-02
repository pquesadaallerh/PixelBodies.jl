import os
import shutil
import time
from pathlib import Path


def ensure_link(src: Path, dst: Path):
    """
    Create a symlink to src at dst. On Windows without privilege, fall back to hardlink or copy.
    """
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception:
            pass
    try:
        dst.symlink_to(src)
        print(f"[link] symlink created: {dst} -> {src}")
        return
    except OSError as e:
        if getattr(e, "winerror", None) == 1314:
            print(f"[link] Symlink privilege missing (1314). Falling back to hardlink/copy.")
        else:
            print(f"[link] Symlink failed ({e}). Trying hardlink/copy.")

    # Hardlink attempt
    try:
        os.link(src, dst)
        print(f"[link] hardlink created: {dst} -> {src}")
        return
    except OSError as e:
        print(f"[link] Hardlink failed ({e}). Copying file.")

    # Copy fallback
    try:
        shutil.copy2(src, dst)
        print(f"[link] file copied: {dst} (from {src})")
    except Exception as e:
        raise RuntimeError(f"Failed to create link or copy from {src} to {dst}: {e}")

def safe_unlink_windows(path: Path, max_retries: int = 3, delay: float = 0.5):
    """
    Simplified unlink - display process will be restarted so no need for complex retry logic.
    """
    for attempt in range(max_retries):
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"[unlink] File locked, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"[unlink] Failed to unlink {path} after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            print(f"[unlink] Unexpected error unlinking {path}: {e}")
            raise
    return False

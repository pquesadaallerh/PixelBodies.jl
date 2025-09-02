from pathlib import Path
import json
import platform
import time

import yaml

from picture_sim_app.characteristic_length_and_aoa_estimation import characteristic_length_and_aoa_pca
from picture_sim_app.detect_airfoil_type import detect_airfoil_type
from picture_sim_app.live_simulation import run_julia_simulation_script
from picture_sim_app.pixel_body_python import PixelBodyMask
from symlink_utils import ensure_link, safe_unlink_windows

# OS-specific imports (some features did not work as expected on a Windows device and required tweaks. In the future
# it would be better to merge the two implementations into one, but due to time constraints there are two parallel
# implementations at the moment).

IS_WINDOWS = platform.system() == "Windows"

# TODO: Merge the two implementations into one
if IS_WINDOWS:
    from picture_sim_app.image_utils_windows import (
        capture_image, stop_display_processes, restart_display_process,
)
else:
    from picture_sim_app.image_utils import (
        capture_image,
    )


# Define absolute path to the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def run_simulation(settings):
    """Run one complete simulation cycle."""
    # Capture image (skips bounding box selection and uses cached option)
    capture_image(
        input_folder=INPUT_FOLDER,
        fixed_aspect_ratio=tuple(settings["capture_image_aspect_ratio"]),  # Aspect ratio of the selection box (width:height)
        selection_box_mode=True,  # Click-and-drag selection box
        # fixed_size=(800, 600),    # Alternative: exact pixel dimensions
        use_cached_box=True,  # Fixed typo: was use_chached_box
        camera_index=settings["io_settings"]["camera_index"],
    )

    # File I/O paths
    io_settings = settings["io_settings"]
    input_path = INPUT_FOLDER / io_settings["input_image_name"]
    output_path_particle_plot = OUTPUT_FOLDER / io_settings["particle_plot_name"]
    output_path_heatmap_vorticity= OUTPUT_FOLDER / io_settings["heatmap_vorticity_name"]
    output_path_heatmap_pressure = OUTPUT_FOLDER / io_settings["heatmap_pressure_name"]
    # output_path_data = OUTPUT_FOLDER / data_file_name
    flip_90_degrees = io_settings["flip_90_degrees"]

    # Unpack simulation settings:
    simulation_settings = settings["simulation_settings"]
    image_recognition_debug_mode = simulation_settings["image_recognition_debug_mode"]
    show_components_pca = simulation_settings["show_components_pca"]

    if show_components_pca and not image_recognition_debug_mode:
        raise Exception("'show_components_pca' is set to True, but 'image_recognition_debug_mode' is False."
                        "\n set 'image_recognition_debug_mode' to True to see PCA components.")

    # Use image recognition to create a fluid-solid mask (1=Fluid, 0=Solid)
    pixel_body = PixelBodyMask(
        image_path=str(input_path),
        threshold=simulation_settings["threshold"],
        diff_threshold=simulation_settings["diff_threshold"],
        max_image_res=simulation_settings["max_image_res"],
        body_color=simulation_settings["solid_color"],
        manual_mode=simulation_settings["manual_mode"],
        force_invert_mask=simulation_settings["force_invert_mask"],
    )
    domain_mask = pixel_body.get_mask()

    plot_mask = image_recognition_debug_mode
    if plot_mask:
        pixel_body.plot_mask()

    object_is_airfoil = True

    if not simulation_settings["use_precomputed_results"]:
        # Estimate characteristic length and angle of attack using PCA
        l_c, aoa, thickness = characteristic_length_and_aoa_pca(
            mask=domain_mask,
            plot_method=image_recognition_debug_mode,
            show_components=show_components_pca,
            object_is_airfoil=object_is_airfoil,
        )

        if object_is_airfoil:
            # Estimate airfoil type based on thickness and characteristic length
            airfoil_type = detect_airfoil_type(thickness_to_cord_ratio=thickness / l_c)

        run_julia_simulation_script(
            domain_mask=domain_mask,
            l_c=l_c,
            simulation_settings=simulation_settings,
            output_path_particle_plot=output_path_particle_plot,
            output_path_heatmap_vorticity=output_path_heatmap_vorticity,
            output_path_heatmap_pressure=output_path_heatmap_pressure,
            output_folder=OUTPUT_FOLDER,
            script_dir=SCRIPT_DIR,
        )

    else:
        # If using precomputed results, try to find the best matching GIF plots and use those instead of running the
        # simulation

        # Estimate characteristic length and angle of attack using PCA and airfoil type detection
        l_c, aoa, thickness = characteristic_length_and_aoa_pca(
            mask=domain_mask,
            plot_method=image_recognition_debug_mode,
            show_components=show_components_pca,
            object_is_airfoil=object_is_airfoil,
        )
        if flip_90_degrees:
            # Rotate the detected angle 90 degrees counterclockwise to match angle if flow is coming from the top
            aoa -= 90

        if object_is_airfoil:
            # Estimate airfoil type based on thickness and characteristic length
            airfoil_type = detect_airfoil_type(thickness_to_cord_ratio=thickness / l_c)

        # Round angle of attack to nearest multiple of 3
        rounded_aoa = round(aoa / 3) * 3

        # Find gifs corresponding to the airfoil type and angle of attack
        particle_plot_name = f"particleplot_{airfoil_type}_{rounded_aoa}.gif"
        heatmap_vorticity_name = f"heatmap_vorticity_{airfoil_type}_{rounded_aoa}.gif"
        heatmap_pressure_name = f"heatmap_pressure_{airfoil_type}_{rounded_aoa}.gif"

        # Check if the plots exists
        output_path_particle_plot = OUTPUT_FOLDER / "batch_runs" / particle_plot_name
        output_path_heatmap_vorticity = OUTPUT_FOLDER / "batch_runs" / heatmap_vorticity_name
        output_path_heatmap_pressure = OUTPUT_FOLDER / "batch_runs" / heatmap_pressure_name

        if not output_path_particle_plot.exists():
            raise FileNotFoundError(f"Could not find {output_path_particle_plot}")

        if not output_path_heatmap_vorticity.exists():
            raise FileNotFoundError(f"Could not find {output_path_heatmap_vorticity}")

        if not output_path_heatmap_pressure.exists():
            raise FileNotFoundError(f"Could not find {output_path_heatmap_pressure}")

        # Overwrite the output paths to the found files (use symlink instead of copying)
        symlink_particle = OUTPUT_FOLDER / "particleplot.gif"
        symlink_heatmap_vorticity = OUTPUT_FOLDER / "heatmap_vorticity.gif"
        symlink_heatmap_pressure = OUTPUT_FOLDER / "heatmap_pressure.gif"

        # If device is Windows, the persistent_gif_display cannot be updated without closing (at least not without
        # overwriting admin permission rights), so we need to stop and restart it.
        if IS_WINDOWS:
            # Stop display processes to release files
            print("Stopping display processes to update symlinks...")
            stop_display_processes()
            time.sleep(1)  # Give processes time to fully stop

            # Remove existing symlinks/files if they exist (should work now)
            safe_unlink_windows(symlink_particle)
            safe_unlink_windows(symlink_heatmap_vorticity)
            safe_unlink_windows(symlink_heatmap_pressure)

            # Create new symlinks pointing to the batch_runs files
            ensure_link(output_path_particle_plot, symlink_particle)
            ensure_link(output_path_heatmap_vorticity, symlink_heatmap_vorticity)
            ensure_link(output_path_heatmap_pressure, symlink_heatmap_pressure)


            # Restart display process
            print("Restarting display process...")
            restart_display_process(
                script_dir=SCRIPT_DIR,
                monitor_index=1,
                use_qt_version=True,
                flip_90_degrees=flip_90_degrees,
            )

        else:
            # Remove existing symlinks/files if they exist
            if symlink_particle.exists() or symlink_particle.is_symlink():
                symlink_particle.unlink()
            if symlink_heatmap_vorticity.exists() or symlink_heatmap_vorticity.is_symlink():
                symlink_heatmap_vorticity.unlink()
            if symlink_heatmap_pressure.exists() or symlink_heatmap_pressure.is_symlink():
                symlink_heatmap_pressure.unlink()

            # Create new symlinks pointing to the batch_runs files
            ensure_link(output_path_particle_plot, symlink_particle)
            ensure_link(output_path_heatmap_vorticity, symlink_heatmap_vorticity)
            ensure_link(output_path_heatmap_pressure, symlink_heatmap_pressure)

    # Save airfoil data to JSON (use actual AoA, not rounded)
    airfoil_data = {
        "airfoil_type": airfoil_type if object_is_airfoil else "unknown",
        "aoa": round(aoa, 1),
        "thickness": round(thickness, 3),
        "characteristic_length": round(l_c, 3)
    }

    airfoil_data_path = OUTPUT_FOLDER / "airfoil_data.json"
    with open(airfoil_data_path, "w") as f:
        json.dump(airfoil_data, f, indent=2)


def main() -> None:
    # Load settings once at start
    with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)

    camera_index = settings["io_settings"]["camera_index"]

    # First run with interactive selection (don't use cached box)
    capture_image(
        input_folder=INPUT_FOLDER,
        fixed_aspect_ratio=tuple(settings["capture_image_aspect_ratio"]),
        selection_box_mode=True,
        use_cached_box=False,  # Interactive selection for first run
        camera_index=camera_index,
    )

    # Run first simulation
    run_simulation(settings)

    # Idle loop waiting for spacebar
    while True:
        user_input = input("\nPress ENTER to run again, 'r' to reselect box, or 'q' to quit: ").strip().lower()
        if user_input == 'q':
            print("Exiting...")
            break
        elif user_input == '':
            print("Running simulation again...")
            # Reload settings in case they changed
            with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            run_simulation(settings)
        elif user_input == 'r':
            print("Reselecting bounding box...")
            capture_image(
                input_folder=INPUT_FOLDER,
                fixed_aspect_ratio=tuple(settings["capture_image_aspect_ratio"]),
                selection_box_mode=True,
                use_cached_box=False,  # Force interactive selection
                camera_index=camera_index,
            )
            # Reload settings and run simulation with new box
            with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            run_simulation(settings)
        else:
            print("Invalid input. Press ENTER, 'r', or 'q'.")


if __name__ == "__main__":
    main()
module PixelBodies

# Include the component files
include("PixelBody.jl")

# Export the main types and functions you want to use
export PixelBody
export create_fluid_solid_mask_using_image_recognition
export pad_to_pow2_with_ghost_cells
export estimate_characteristic_length
export characteristic_length_bbox
export characteristic_length_pca
export calculate_angle_of_attack
export limit_resolution

end # module

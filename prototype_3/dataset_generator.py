import os
import cv2
import numpy as np
import random
import math

# --- Configuration ---
IMG_SIZE = 64
NUM_IMAGES = 200
OUTPUT_DIR = "datasets/mri_dataset_v2" # Updated version for new shapes
NOISE_SIGMA_RANGE = (5, 8)

# --- Helper Functions ---

def create_directories():
    """Creates the necessary folder structure for the dataset."""
    # Updated to create directories for 5 contrasts
    for i in range(1, 6):
        os.makedirs(os.path.join(OUTPUT_DIR, f"contrast_{i}", "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, f"contrast_{i}", "masks"), exist_ok=True)
    print(f"✅ Dataset directories created in '{OUTPUT_DIR}'")

def add_gaussian_noise(image, sigma):
    """Adds Gaussian noise to a uint8 image."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def is_point_in_ellipse(point, center, axes, angle):
    """Check if a point is inside a rotated ellipse."""
    px, py = point; cx, cy = center; ax, ay = axes
    theta = -np.deg2rad(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    x_rot = (px - cx) * cos_a - (py - cy) * sin_a
    y_rot = (px - cx) * sin_a + (py - cy) * cos_a
    return (x_rot**2 / ax**2) + (y_rot**2 / ay**2) <= 1.0

def generate_random_point_in_ellipse(center, axes, angle):
    """Generates a random point guaranteed to be inside a rotated ellipse."""
    while True:
        max_r = max(axes)
        point = (random.uniform(center[0] - max_r, center[0] + max_r),
                 random.uniform(center[1] - max_r, center[1] + max_r))
        if is_point_in_ellipse(point, center, axes, angle):
            return tuple(map(int, point))

def draw_ellipse(image, center, axes, angle, color, thickness):
    """A wrapper for cv2.ellipse that handles integer conversion."""
    cv2.ellipse(image, tuple(map(int, center)), tuple(map(int, axes)), int(angle), 0, 360, int(color), int(thickness))

def get_rotated_poly_points(center, points, angle_deg):
    """Rotates polygon points around a center and returns them."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    new_points = []
    for p in points:
        x, y = p[0], p[1]
        new_x = x * cos_a - y * sin_a + center[0]
        new_y = x * sin_a + y * cos_a + center[1]
        new_points.append([int(new_x), int(new_y)])
    return np.array(new_points, dtype=np.int32)

def draw_complex_lesion(image, lesion_params, color):
    """Draws a complex lesion made of an ellipse, triangle, and rectangle."""
    # 1. Draw the base ellipse
    draw_ellipse(image, lesion_params['center'], lesion_params['ellipse_axes'],
                 lesion_params['ellipse_angle'], color, -1)

    # 2. Define and draw the rotated triangle
    h = lesion_params['tri_size']
    tri_points_origin = np.array([[0, -h//1.7], [-h//1.7, h//1.7], [h//1.7, h//1.7]])
    rotated_tri_points = get_rotated_poly_points(lesion_params['center'], tri_points_origin, lesion_params['tri_angle'])
    cv2.fillPoly(image, [rotated_tri_points], color)

    # 3. Define and draw the rotated rectangle
    w, h = lesion_params['rect_dims']
    rect_points_origin = np.array([[-w//3, -h//3], [w//1.7, -h//1.7], [w//1.7, h//1.7], [-w//1.7, h//1.7]])
    rotated_rect_points = get_rotated_poly_points(lesion_params['center'], rect_points_origin, lesion_params['rect_angle'])
    cv2.fillPoly(image, [rotated_rect_points], color)

# --- Main Generation Function ---

def generate_dataset():
    """Generates and saves the complete synthetic dataset."""
    create_directories()

    for i in range(NUM_IMAGES):
        # 1. Define background parameters
        center = (IMG_SIZE // 2, IMG_SIZE // 2)
        outer_axes = (random.randint(18, 22), random.randint(24, 28))
        outer_angle = random.uniform(-15, 15)
        outer_intensity = random.randint(230, 255)
        inner_axes = (outer_axes[0] - 8, outer_axes[1] - 8)
        inner_angle = outer_angle + random.uniform(-5, 5)
        inner_intensity_default = random.randint(115, 165)
        inner_intensity_contrast2 = random.randint(77, 89)

        # 2. Define parameters for two complex lesions
        lesions = []
        for _ in range(2):
            params = {
                'center': generate_random_point_in_ellipse(center, (inner_axes[0]-7, inner_axes[1]-7), inner_angle),
                'intensity': random.randint(64, 77),
                'ellipse_axes': (random.randint(3, 5), random.randint(4, 6)),
                'ellipse_angle': random.uniform(0, 180),
                'tri_size': random.randint(8, 12),
                'tri_angle': random.uniform(0, 180),
                'rect_dims': (random.randint(6, 10), random.randint(4, 7)),
                'rect_angle': random.uniform(0, 180)
            }
            lesions.append(params)

        # 3. Create the Ground Truth Mask
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        for lesion in lesions:
            draw_complex_lesion(mask, lesion, 255) # Draw shape in white

        # 4. Generate each contrast image
        base_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        draw_ellipse(base_img, center, outer_axes, outer_angle, outer_intensity, 2)
        draw_ellipse(base_img, center, inner_axes, inner_angle, inner_intensity_default, -1)

        # Contrast 3: Perfect
        img_c3 = base_img.copy()
        for lesion in lesions:
            draw_complex_lesion(img_c3, lesion, lesion['intensity'])

        # --- Generate Contrast 1 and 5 together ---
        # Contrast 1: Occluded (lesion with holes)
        # Contrast 5: Complementary (only the holes are visible, on the standard background)
        img_c1 = img_c3.copy()
        img_c5 = base_img.copy() # CORRECTED: Start with the standard background, not a black image

        for lesion in lesions:
            # Generate occlusions within the base ellipse of the lesion
            for _ in range(random.randint(5, 11)):
                occlusion_center = generate_random_point_in_ellipse(lesion['center'], lesion['ellipse_axes'], lesion['ellipse_angle'])
                occlusion_axes = (random.randint(1, 3), random.randint(1, 3))

                # On C1, draw the occlusion with the background color to "hide" a part of the lesion
                draw_ellipse(img_c1, occlusion_center, occlusion_axes, 0, inner_intensity_default, -1)

                # On C5, draw the same occlusion with the lesion's color to "reveal" the missing part
                draw_ellipse(img_c5, occlusion_center, occlusion_axes, 0, lesion['intensity'], -1)

        # Contrast 2: Low Contrast
        img_c2 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        draw_ellipse(img_c2, center, outer_axes, outer_angle, outer_intensity, 2)
        draw_ellipse(img_c2, center, inner_axes, inner_angle, inner_intensity_contrast2, -1)
        for lesion in lesions:
            draw_complex_lesion(img_c2, lesion, lesion['intensity'])

        # Contrast 4: Impossible (lesion is not visible)
        img_c4 = base_img.copy()

        # 5. Add noise and save all 5 contrasts
        noise_sigma = random.uniform(NOISE_SIGMA_RANGE[0], NOISE_SIGMA_RANGE[1])
        all_contrasts = {1: img_c1, 2: img_c2, 3: img_c3, 4: img_c4, 5: img_c5}
        for idx, img in all_contrasts.items():
            noisy_img = add_gaussian_noise(img, sigma=noise_sigma)
            img_path = os.path.join(OUTPUT_DIR, f"contrast_{idx}", "images", f"{i:04d}.png")
            mask_path = os.path.join(OUTPUT_DIR, f"contrast_{idx}", "masks", f"{i:04d}.png")
            cv2.imwrite(img_path, noisy_img)
            cv2.imwrite(mask_path, mask)

        if (i + 1) % 25 == 0:
            print(f"  ... Generated {i+1}/{NUM_IMAGES} image sets")

    print("🚀 Dataset generation complete!")

if __name__ == "__main__":
    generate_dataset()

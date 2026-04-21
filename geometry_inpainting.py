import cv2
import numpy as np

def generate_component_mask(gray_image, bbox):
    """
    Given a bounding box, dynamically segment the black ink (the strikeout) from the gray image
    using Otsu's thresholding to recreate the exact pixel mask footprint of the component.
    """
    x, y, w, h = bbox
    # Boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(gray_image.shape[1], x + w), min(gray_image.shape[0], y + h)
    
    crop = gray_image[y1:y2, x1:x2]
    # Apply Otsu's to find ink (ink is usually dark/black). 
    # cv2.THRESH_BINARY_INV makes ink = 255, background = 0
    if crop.size > 0:
        _, binary_mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary_mask = np.zeros_like(crop)
    
    return binary_mask, (x1, y1, x2, y2)

def apply_boundary_feathering(crop, mask, d=5):
    """
    Implements MathScrub Eq. 10:
    I(x,y) = (1 - alpha)*I_b(x,y) + alpha * I_white
    alpha = D(x,y) / d  (clipped to [0,1])
    """
    # 1. Distance transform from the boundary of the strikeout mask.
    # cv2.distanceTransform computes distance from zero pixels to nearest zero pixel.
    # Here, mask=255 is the strikeout. We want distance from the INSIDE of the strikeout to its boundary.
    # So we invert the mask, making the strikeout 0, BUT we want the edge.
    # Let's find the inner distance. 
    # Distance transform on the strikeout mask (ink=255) gives distance of ink pixels to the nearest background pixel.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    
    # 2. Calculate alpha = dist / d
    alpha = dist / float(d)
    np.clip(alpha, 0.0, 1.0, out=alpha)
    
    # 3. I_white is 255
    I_white = np.full_like(crop, 255, dtype=np.float32)
    
    # 4. Applying the equation:
    # Here I_b(x,y) ideally should be the boundary pixel extended inward, but for simplicity
    # and reading the Eq, I(x,y) blends the actual crop pixel with white as it goes deeper.
    # To truly erase, we blend the existing grayscale pixel (which is dark ink) with white.
    # As alpha -> 1 (deep inside), it becomes 100% white.
    # As alpha -> 0 (on the boundary), it remains 100% the original boundary pixel I_b.
    I_new = (1.0 - alpha) * crop.astype(np.float32) + alpha * I_white
    
    # Only modify the pixels that belong to the strikeout mask
    result = crop.copy()
    mask_indices = mask > 0
    result[mask_indices] = I_new[mask_indices].astype(np.uint8)
    
    return result

def inpaint_math_equation(gray_image, components, predictions, tau_p=0.5, T=15316, d=5):
    """
    Orchestrating Phase 3 logic.
    - Filters predictions to find strikeouts (p >= tau_p)
    - If Area <= T: Navier-Stokes interpolation
    - If Area > T: Boundary-aware feathering Repasting
    """
    restored_img = gray_image.copy()
    
    # Create an empty global mask to collect small strokes for Navier Stokes
    global_ns_mask = np.zeros_like(gray_image)
    
    for comp, pred in zip(components, predictions):
        # Safety check matching
        if comp["id"] != pred["component_id"]:
            continue
            
        p_delete = pred["p_delete"]
        if p_delete >= tau_p: # This is a strikeout
            area = comp.get("area", 0)
            bbox = comp.get("bbox", [0,0,0,0])
            
            # Retrieve exact pixel mask footprint
            local_mask, (x1, y1, x2, y2) = generate_component_mask(gray_image, bbox)
            
            if area <= T:
                # Add to Navier-Stokes global mask
                global_ns_mask[y1:y2, x1:x2] = cv2.bitwise_or(global_ns_mask[y1:y2, x1:x2], local_mask)
            else:
                # Apply Boundary Feathering locally immediately
                crop = restored_img[y1:y2, x1:x2]
                feathered_crop = apply_boundary_feathering(crop, local_mask, d=d)
                restored_img[y1:y2, x1:x2] = feathered_crop
                
    # Once massive occlusions are feathered via iteration, we run OpenCV's highly optimized 
    # Navier-Stokes on the entire remaining global mask of small gaps at once!
    if np.any(global_ns_mask > 0):
        # inpaintRadius matches 'd' roughly
        restored_img = cv2.inpaint(restored_img, global_ns_mask, inpaintRadius=d, flags=cv2.INPAINT_NS)
        
    return restored_img

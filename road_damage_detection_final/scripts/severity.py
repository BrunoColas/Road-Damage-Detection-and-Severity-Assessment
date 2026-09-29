CLASS_WEIGHT = {
    "Longitudinal crack": 0.90,
    "Transverse crack": 0.90,
    "Alligator crack": 1.15,
    "Pothole": 1.20,
}

def estimate_severity(class_name, x1, y1, x2, y2, image_w, image_h):
    image_area = max(image_w * image_h, 1)
    box_area = max((x2 - x1) * (y2 - y1), 1)
    area_ratio = box_area / image_area

    wr = max((x2 - x1) / image_w, 1e-6)
    hr = max((y2 - y1) / image_h, 1e-6)
    elongation = max(wr / hr, hr / wr)

    # Geometry contributes modestly; area is the dominant signal.
    geometry_factor = min(1.40, 1.0 + 0.08 * max(0, elongation - 3))
    weight = CLASS_WEIGHT.get(class_name, 1.0)

    score = min(1.0, (area_ratio ** 0.5) * 3.0 * weight * geometry_factor)

    if score < 0.18:
        severity = "Low"
    elif score < 0.42:
        severity = "Medium"
    else:
        severity = "High"

    return severity, round(float(score), 4)

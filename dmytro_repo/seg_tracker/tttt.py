import config
print(config.DETECTOR_ONLY)

detected_objects = {}
detect_objects['x'] = [1, 2, 3, 4, 5, 6]

    detected_items = [
        DetectedItem(
            x = 5
        )
        for it in detected_objects
        if it['x'] >= 3
    ]
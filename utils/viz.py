import cv2

def draw_bboxes(image, bboxes, labels):
    # bbox in format [left, width, top, height]
    for i, bbox in enumerate(bboxes):
        (x, y) = (bbox[0], bbox[2])
        (w, h) = (bbox[1], bbox[3])

        image = cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 255),
                    thickness=2
                )
    return image
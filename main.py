import detector
import cv2
import numpy as np

model = 'data/best_ckpt.onnx'
data_yaml = 'data/data.yaml'
obj = detector.YOLOv6(model, data_yaml)

vid = cv2.VideoCapture(0)

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

out = cv2.VideoWriter('test_dice_ai.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (frame_width, frame_height))

if __name__ == '__main__':

    while True:
        # Capture frame-by-frame
        ret, frame = vid.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # sharpened = cv2.filter2D(frame, -1, filter)

        obj.detect_objects(frame)
        drawn = obj.draw_detections(frame)

        out.write(drawn)
        cv2.imshow('frame', drawn)

        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    out.release()

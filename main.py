import cv2
import pandas as pd
import os
import numpy as np
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class PosePredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'pose'

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        0.7,
                                        0.55,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results


def IOU(box1, box2):
    # Coodinates of the intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # Area of overlap - width + height
    width = (x2 - x1)
    height = (y2 - y1)
    if width < 0 or height < 0:
        return 0.0
    area_overlap = width * height
    # Area combined
    Area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    Area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    Area_combined = Area1 + Area2 - area_overlap
    IoU = area_overlap / Area_combined
    return IoU


predictor = PosePredictor(overrides=dict(model='yolov8x-pose-p6.pt'))

src_video = 'videos/stand02_5.mp4'
src_name = src_video.split('/')[-1].split('.')[0]
os.mkdir(f"rgb_image/{src_name}")
cap = cv2.VideoCapture(src_video)
print(cap.isOpened())

frame_count = 0

output_path = f"{src_name}_outputv8_v.mp4"
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)),
                               (int(cap.get(3)), int(cap.get(4))))

list_objects = pd.DataFrame(columns=['id', 'bb_xyxy', 'xyn_skeleton', 'rgb', 'frame'])
base_quantity = 0
compare_window = []
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # Run pose detection on the
    # Resize the image to the desired size using cv2.resize()
    if success:
        frame_count += 1
        print(f"Frame {frame_count}")
        results1 = predictor(frame)
        res = results1[0]
        # Visualize the results on the frame
        annotated_frame = results1[0].plot()
        n = len(results1[0])
        if frame_count == 1:
            for i in range(n):
                os.mkdir(f'rgb_image/{src_name}/id_{i+1}')
        index_max, max_iou_score = 0, [0, 0, 0, 0]
        for idx in range(n):
            id_person = idx + 1
            kpt_list = res.keypoints.xy.cpu().numpy().tolist()[idx][:13]
            kptN_list = res.keypoints.xyn.cpu().numpy().tolist()[idx][:13]
            bbox_coordinates = res.boxes.xyxy.cpu().numpy().tolist()[idx]
            kpt_x, kpt_y = 0, 0
            for kpt in kpt_list:
                if kpt[0] and kpt[1]:
                    kpt_x, kpt_y = kpt[0], kpt[1]
                    break
            bb_x, bb_y = bbox_coordinates[0], bbox_coordinates[1]
            rgb_image = frame[int(bbox_coordinates[1]):int(bbox_coordinates[3]), int(bbox_coordinates[0]):int(bbox_coordinates[2])]
            new_array_update = [id_person, bbox_coordinates, kptN_list, rgb_image, f"frame{frame_count}"]
            new_dataframe = {'id': id_person, 'bb_xyxy': bbox_coordinates, 'xyn_skeleton': kptN_list,
                             'rgb': rgb_image, 'frame': f"frame{frame_count}"}
            if frame_count == 1:
                base_quantity = n
                list_objects.loc[-1] = new_dataframe
                list_objects.index += 1
                if kpt_x and kpt_y:
                    annotated_frame = cv2.putText(annotated_frame, f"{id_person}", (int(kpt_x), int(kpt_y)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (56, 56, 255), 2, cv2.LINE_AA)
                if bb_x and bb_y:
                    annotated_frame = cv2.putText(annotated_frame, f"{id_person}", (int(bb_x), int(bb_y)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (56, 56, 255), 2, cv2.LINE_AA)
            else:
                changed = False
                for i in range(len(compare_window)):
                    iou_score = IOU(compare_window[i][1], bbox_coordinates)
                    if iou_score > max_iou_score[i] and iou_score >= 0.8:
                        max_iou_score[i] = iou_score
                        index_max = i
                        new_array_update[0] = compare_window[i][0]
                        compare_window[index_max] = new_array_update
                        changed = True
                if kpt_x and kpt_y and changed:
                    annotated_frame = cv2.putText(annotated_frame, f"{index_max+1}", (int(kpt_x), int(kpt_y)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (56, 56, 255), 2, cv2.LINE_AA)
                if bb_x and bb_y and changed:
                    annotated_frame = cv2.putText(annotated_frame, f"{index_max+1}", (int(bb_x), int(bb_y)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (56, 56, 255), 2, cv2.LINE_AA)
        for value in compare_window:
            if frame_count != 1:
                new_frame = {'id': value[0], 'bb_xyxy': value[1], 'xyn_skeleton': value[2], 'rgb': value[3], 'frame': value[4]}
                list_objects.loc[-1] = new_frame
                list_objects.index += 1
            if frame_count != 1:
                compare_window = []
        for _, row in list_objects.tail(base_quantity).iterrows():
            row_values = [list(val) if isinstance(val, list) else val for val in row.values.flatten()]
            compare_window.append(row_values)
        for value in compare_window:
            output_rgb = f"rgb_image/{src_name}/id_{value[0]}/{src_name}_rgb_{value[0]}_{value[4]}.png"
            cv2.imwrite(output_rgb, value[3])
        video_writer.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    csv_save_path = f"{src_video.split('/')[-1].split('.')[0]}.csv"
    list_objects.to_csv(csv_save_path, index=False)
video_writer.release()
cap.release()

import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 基础参数
PINCH_THRESH = 30
brush_color = (200, 150, 0)
rect_line_color = (200, 150, 180)
brush_thickness = 6

# 绘图状态
draw_last_point = None
draw_canvas = None
dynamic_rect = None

# 初始化 MediaPipe 手部检测器（新版 Tasks API）
model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
if not os.path.exists(model_path):
    print(f"模型文件不存在: {model_path}")
    print("请先下载 hand_landmarker.task")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, temp_frame = cap.read()
if not ret:
    print("摄像头打开失败")
    cap.release()
    cv2.destroyAllWindows()
    exit()

img_h, img_w, _ = temp_frame.shape
draw_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)


def clamp(val, min_v, max_v):
    """坐标边界限制，防止 ROI 切片越界闪退"""
    return max(min_v, min(val, max_v))


frame_idx = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1
    frame = cv2.flip(frame, 1)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_white_bg = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # 新版 API：将帧转为 MediaPipe Image 再检测
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_result = detector.detect_for_video(mp_image, frame_idx)

    hand_info = []
    dynamic_rect = None

    # 读取双手关键点（拇指尖 4，食指尖 8）
    if hand_result.hand_landmarks:
        for hand_lm in hand_result.hand_landmarks:
            thumb_pt = hand_lm[4]
            index_pt = hand_lm[8]
            tx = int(thumb_pt.x * img_w)
            ty = int(thumb_pt.y * img_h)
            ix = int(index_pt.x * img_w)
            iy = int(index_pt.y * img_h)
            hand_info.append((tx, ty, ix, iy))

    # 双手：动态跟随彩色矩形（框内彩色、框外灰度）
    if len(hand_info) == 2:
        draw_last_point = None
        h1 = hand_info[0]
        h2 = hand_info[1]
        all_x = [h1[0], h1[2], h2[0], h2[2]]
        all_y = [h1[1], h1[3], h2[1], h2[3]]

        x_min = clamp(min(all_x), 0, img_w - 1)
        y_min = clamp(min(all_y), 0, img_h - 1)
        x_max = clamp(max(all_x), 0, img_w - 1)
        y_max = clamp(max(all_y), 0, img_h - 1)

        if x_max > x_min and y_max > y_min:
            dynamic_rect = [(x_min, y_min), (x_max, y_max)]

    # 单手：捏合手绘，轨迹保留
    elif len(hand_info) == 1:
        dynamic_rect = None
        tx, ty, ix, iy = hand_info[0]
        pinch_dis = np.hypot(tx - ix, ty - iy)
        current_draw_pt = (ix, iy)

        if pinch_dis < PINCH_THRESH:
            if draw_last_point is not None:
                cv2.line(draw_canvas, draw_last_point, current_draw_pt, brush_color, brush_thickness)
            draw_last_point = current_draw_pt
        else:
            draw_last_point = None
    else:
        draw_last_point = None
        dynamic_rect = None

    # 叠加手绘线条到灰度背景
    base_img = cv2.addWeighted(black_white_bg, 1, draw_canvas, 1, 0)
    final_img = base_img.copy()

    # 双手模式下：框内区域还原为彩色
    if dynamic_rect is not None:
        pt1, pt2 = dynamic_rect
        cv2.rectangle(final_img, pt1, pt2, rect_line_color, 6)
        roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        final_img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = roi

    cv2.imshow("Gesture Program", final_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        draw_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()

#!/usr/bin/python3
# 导入必要的库
from imutils import contours
from skimage import measure
import numpy as np
import cv2

CAM = 0  # 摄像头索引，您可以根据需要修改
cap = cv2.VideoCapture(CAM)

def nothing(x):
    pass

# 创建参数控制窗口
cv2.namedWindow("params")
cv2.createTrackbar("manual", "params", 0, 1, nothing)
cv2.createTrackbar("pixels_num", "params", 0, 2000, nothing)

ret, frame = cap.read()
if not ret:
    print("无法读取摄像头，请检查设备连接。")
    exit()

H, W = frame.shape[:2]  # 获取帧的高和宽

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧。")
        break

    image = frame.copy()
    # 将图像转换为灰度图并进行高斯模糊
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 对图像进行阈值处理，突出亮点（激光点）
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    # 进行形态学操作，去除噪声
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # 连通区域分析
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # 获取Trackbar参数
    MANUAL = cv2.getTrackbarPos("manual", "params")
    PIXEL_THRESH = cv2.getTrackbarPos("pixels_num", "params") if MANUAL == 1 else 500

    # 遍历每个连通区域
    for label in np.unique(labels):
        if label == 0:
            continue
        # 创建标签掩膜并计算像素数
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # 显示像素数量（可选）
        cv2.putText(image, "pixel_num:{}".format(numPixels), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 根据像素数量阈值过滤
        if numPixels < PIXEL_THRESH:
            mask = cv2.add(mask, labelMask)

    # 查找轮廓
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # 兼容不同OpenCV版本

    if len(cnts) == 0:
        print("未检测到轮廓")
        cv2.imshow("mask", mask)
        cv2.imshow("image", image)
        k = cv2.waitKey(40) & 0xff
        if k == 27:
            break
        continue

    # 只保留最大轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # 遍历轮廓
    for (i, c) in enumerate(cnts):
        # 绘制外接矩形和圆形
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 255, 0), 3)
        cv2.putText(image, "#{} at ({},{})".format(i + 1, int(cX), int(cY)), (int(cX), int(cY) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        break  # 只处理最大的轮廓

    # 显示结果
    cv2.imshow("mask", mask)
    cv2.imshow("image", image)
    k = cv2.waitKey(40) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

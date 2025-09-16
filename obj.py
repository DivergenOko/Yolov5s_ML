import random
import torch
import cv2
import numpy as np

with open("utils/coco.txt", "r") as с:
    class_list = с.read().split("\n")

cup_index = -1
for i, cls in enumerate(class_list):
    if cls.lower() in ["cup"]:
        cup_index = i
        break

detection_colors = []
for i in range(len(class_list)):
    if i == cup_index:
        detection_colors.append((0, 255, 0))
    else:
        while True:
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            if (r, g, b) != (0, 255, 0):
                detection_colors.append((b, g, r))
                break


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Камера не работает")
    exit()

print("Нажми 'q' для выхода")

while True:
    ret, frame = cap.read()
    if ret == False:
        print("кадр не захвачен")
        break

    results = model(frame)

    w_frame = 640
    h_frame = 480
    c_frame = (w_frame // 2, h_frame // 2)

    for detection in results.box_xy[0]:
        x1, y1, x2, y2 = map(int, detection[:4])
        conf = float(detection[4])
        index_classYolo = int(detection[5])

        if index_classYolo < len(detection_colors):
            cv2.rectangle(frame, (x1, y1), (x2, y2), detection_colors[index_classYolo], 10) # рамка-обводка
            cv2.putText(frame, f"{class_list[index_classYolo]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 5) # класс+точность над рамкой

            if index_classYolo == cup_index:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                c_cup = (cx, cy)
                print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
                print(f"Центр Cup: ({cx}, {cy})")

                dx = cx - c_frame[0]
                dy = cy - c_frame[1]
                d = ((dx ** 2 + dy ** 2) ** 0.5)
                print(f"Вектор: ({dx}, {dy})")
                if d != 0:
                    ux = dx / d
                    uy = dy / d
                    print(f"Норм. вектор: ({ux:.2f}, {uy:.2f})")
                else:
                    print(f"Норм. вектор: (0.00, 0.00)")
                print(f"Расстояние: {d:.2f}")

                cv2.circle(frame, c_cup, 5, (0, 0, 255), -1)  # Красная точка
                cv2.circle(frame, c_frame, 5, (255, 0, 0), -1)  # Синяя точка
                cv2.arrowedLine(frame, c_frame, c_cup, (255, 0, 255), 2)  # Фиолетовая стрелка

    cv2.imshow("Детекция", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Выход")
        break

cap.release()
cv2.destroyAllWindows()
print("Камера выключена")
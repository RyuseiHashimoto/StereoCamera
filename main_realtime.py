import numpy as np
import cv2
import pickle

T = 70  # カメラ間距離 (mm)

# キャリブレーション結果を読み込む
with open('calibration_720/calibration_720.pkl', 'rb') as f:
    cameraMatrix_720, dist_720 = pickle.load(f)
with open('calibration_HD720/calibration_HD720.pkl', 'rb') as f:
    cameraMatrix_HD720, dist_HD720 = pickle.load(f)

# 焦点距離を取得
F1 = cameraMatrix_720[0, 0]  # 720カメラのx方向の焦点距離（px）
F2 = cameraMatrix_HD720[0, 0]  # HD720カメラのx方向の焦点距離（px）

# ウェブカメラを開く
cap_HD720 = cv2.VideoCapture(0)
cap_720 = cv2.VideoCapture(1)

if not cap_720.isOpened() or not cap_HD720.isOpened():
    print("Error: Could not open the webcam.")
    exit()


# メインプロセス
while True:
    # フレーム読み込み
    ret_720, frame_720 = cap_720.read()
    ret_HD720, frame_HD720 = cap_HD720.read()

    if not ret_720 or not ret_HD720:
        print("Failed to capture frame. Exiting...")
        break

    # フレーム補正
    undistorted_720 = cv2.undistort(frame_720, cameraMatrix_720, dist_720, None)
    undistorted_HD720 = cv2.undistort(frame_HD720, cameraMatrix_HD720, dist_HD720, None)

    # 画像をHSV色空間に変換
    hsv1 = cv2.cvtColor(undistorted_720, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(undistorted_HD720, cv2.COLOR_BGR2HSV)

    # [色相(H),彩度(S),明度(V)]の下限、上限を指定(H:0~180, S:0~255, V:0~255)
    lower_color = np.array([35, 100, 100])
    upper_color = np.array([85, 255, 255])
    mask1 = cv2.inRange(hsv1, lower_color, upper_color)
    mask2 = cv2.inRange(hsv2, lower_color, upper_color)

    # 重心を取得
    M1 = cv2.moments(mask1)
    M2 = cv2.moments(mask2)
    if M1["m00"] != 0 and M2["m00"] != 0:
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])

        # 重心間のピクセル誤差と距離計算
        D = abs(cx1 - cx2)
        if D != 0:
            z1 = (F1 * T) / D
            z2 = (F2 * T) / D
            z = (z1 + z2) / 2
            print(f"推定距離: {z:.2f} mm")

        else:
            print("重心位置の差がゼロです。距離を計算できません。")
    else:
        print("重心が検出されなかったため、距離を計算できませんでした。")

    # 補正フレームを表示
    cv2.imshow('720 Frame', undistorted_720)
    cv2.imshow('HD720 Frame', undistorted_HD720)

    # "q"キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("プログラム終了")
        break


# リソース解放
cap_720.release()
cap_HD720.release()
cv2.destroyAllWindows()

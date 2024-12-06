import numpy as np
import cv2
import pickle

T = 70  # カメラ間距離 (mm)
L = 1000 # 地面からカメラまでの距離(mm)


#キャリブレーション結果(calibration.pkl)を読み込む
with open('calibration_720/calibration_720.pkl', 'rb') as f:
    cameraMatrix_720, dist_720 = pickle.load(f)
with open('calibration_HD720/calibration_HD720.pkl', 'rb') as f:
    cameraMatrix_HD720, dist_HD720 = pickle.load(f)

# 焦点距離を取得
F1 = cameraMatrix_720[0, 0]  # 720カメラのx方向の焦点距離（px）
F2 = cameraMatrix_HD720[0, 0]  # HD720カメラのx方向の焦点距離（px）

# ウェブカメラを開く
cap_720 = cv2.VideoCapture(1)
cap_HD720 = cv2.VideoCapture(0)

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

    # フレームサイズを取得
    h, w = frame_720.shape[:2]

    # フレームを補正
    undistorted_720 = cv2.undistort(frame_720, cameraMatrix_720, dist_720, None)
    undistorted_HD720 = cv2.undistort(frame_HD720, cameraMatrix_HD720, dist_HD720, None)

    # 元フレーム表示
    #cv2.imshow('Original 720 Frame', frame_720)
    #cv2.imshow('Original HD720 Frame', frame_HD720)

    # 補正フレーム表示
    cv2.imshow('Undistorted 720 Frame', undistorted_720)
    cv2.imshow('Undistorted HD720 Frame', undistorted_HD720)

    # キー操作の処理
    key = cv2.waitKey(1) & 0xFF

    # "c"キーで左右のカメラの画像を保存
    if key == ord('c'):
        cv2.imwrite('720_image.jpg', undistorted_720)
        cv2.imwrite('HD720_image.jpg', undistorted_HD720)
        print("左右カメラ画像を保存しました")
    
    # "e"キーで距離計測
    if key == ord('e'):
        # 左右のカメラで撮影された画像を読み込む
        image = cv2.imread('720_image.jpg')
        HD_image = cv2.imread('HD720_image.jpg')
        
        # 画像をHSV色空間に変換
        hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(HD_image, cv2.COLOR_BGR2HSV)

        # フレームのフィルタリングとmask生成 (緑色を抽出)
        lower_colore = np.array([35, 100, 70])
        upper_colore = np.array([85, 255, 255])
        mask1 = cv2.inRange(hsv1, lower_colore, upper_colore)
        mask2 = cv2.inRange(hsv2, lower_colore, upper_colore)

        # 画像1の重心探索
        M1 = cv2.moments(mask1)
        if M1["m00"] != 0:
            cx1 = int(M1["m10"] / M1["m00"])
            cy1 = int(M1["m01"] / M1["m00"])
            pt1 = (cx1, cy1)
            cv2.circle(mask1, pt1, 10, (100), 3)
        else:
            pt1 = None

        # 画像2の重心探索
        M2 = cv2.moments(mask2)
        if M2["m00"] != 0:
            cx2 = int(M2["m10"] / M2["m00"])
            cy2 = int(M2["m01"] / M2["m00"])
            pt2 = (cx2, cy2)
            cv2.circle(mask2, pt2, 10, (100), 3)
        else:
            pt2 = None

        # 画像表示
        cv2.imshow("720_mask_image", mask1)
        cv2.imshow("light_mask_image", mask2)

        # 距離の計算
        if pt1 is not None and pt2 is not None:
            D = abs(pt1[0] - pt2[0])  # ピクセル誤差
            if D != 0:
                z1 = (F1 * T) / D  # 推定距離1
                z2 = (F2 * T) / D  # 推定距離2
                z  = (z1+ z2) / 2  # 平均推定距離
                print(f"推定距離: {z:.2f} mm")
                l  = L - z # 草丈算出
                print(f"草丈: {l:.2f} mm")

            else:
                print("重心位置の差がゼロです。距離を計算できません。")
        else:
            print("重心が検出されなかったため、距離を計算できませんでした。")
        
    # "q"キーで終了
    if key == ord('q'):
        print("プログラム終了")
        break


# Release the webcam and close all windows
cap_720.release()
cap_HD720.release()
cv2.destroyAllWindows()

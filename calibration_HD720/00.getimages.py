import cv2
import pickle

def main():
    cap = cv2.VideoCapture(0)

    # Get frame size (width and height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameSize = (width, height)

    # Save frame size to a file
    with open('calibration_HD720/frameSize_HD720.pkl', 'wb') as f:
        pickle.dump(frameSize, f)

    num = 0
    while cap.isOpened():
        success, img = cap.read()
        k = cv2.waitKey(5)
        if k == 27 or  k == ord('q'):
            break
        elif k == ord('c'): 
            cv2.imwrite('calibration_HD720/images/img' + str(num) + '.png', img)
            num += 1
        cv2.imshow("Press S to Capture images", img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

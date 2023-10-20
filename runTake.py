
import cv2

from runPreProcesing import createTxt
from runPreProcesing import convertImg

def camera():

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Take a Photo")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Take a Photo", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "output/rec/photo{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # camera()
    
    createTxt('output/rec/')
    
    # convertImg()
    
    
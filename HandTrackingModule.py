import cv2 # OpenCV library for video capture and image processing.
import mediapipe as mp # Mediapipe library for hand detection and tracking.
import time 

class HandDetector: #class
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): #handtracking mode, max num of hands, detection confidence, tracking confidence
        self.mode = mode #input images as static images batch or no (taken as argument in init)
        self.maxHands = maxHands #max num of hands taken as argument in init
        self.detectionCon = detectionCon #max confidence taken as argument in init
        self.trackCon = trackCon #max confidence taken as argument in init

        self.mpHands = mp.solutions.hands #Mediapipe module used as class func
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon) #initialized with arguments (self)
        self.mpDraw = mp.solutions.drawing_utils #Mediapipe Drawer used as class func

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #bgr to rgb as Mediapipe needs rgb colors
        self.results = self.hands.process(imgRGB) #processes image to find landmarks (video is basically tons of images)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #if landmarks found in image, draw on them
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks: #if landmarks exist
            myHand = self.results.multi_hand_landmarks[handNo] 
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape #height, width and channels of the image
                cx, cy = int(lm.x * w), int(lm.y * h) #landmark coordinates to pixel coordinates on screen
                lmList.append([id, cx, cy]) #landmarks list appends cx, cy and id
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) #draw circle on the landmark positions
        return lmList

def main():
    pTime = 0 #Previous time # calculate FPS.
    cap = cv2.VideoCapture(0) #Capture vid from default webcam
    detector = HandDetector() #call class through this variable

    while True:
        success, img = cap.read() #read the images from the webcam
        if not success: #if cant read, then break
            break

        img = detector.findHands(img) #class function to locate hands
        lmList = detector.findPosition(img) #class function to find postiion of hands
        if lmList:
            print(lmList[4]) #prints landmarks

        cTime = time.time() #current time
        fps = 1 / (cTime - pTime) #fps calculation
        pTime = cTime #previous time updated

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3) #prints FPS: how much fps there is rn

        cv2.imshow("Image", img) #shows each image as it reads from the webcam
        if cv2.waitKey(1) & 0xFF == ord('q'): #quit key is q
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

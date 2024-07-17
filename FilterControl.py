import cv2
import mediapipe as mp
import time
import os
import numpy as np
from HandTrackingModule import HandDetector  # Import the HandDetector class

class FilterControl:
    def __init__(self): #class
        self.detector = HandDetector() #call HandDetector class
        self.filters = [self.no_filter, self.grayscale, self.sepia, self.negative, self.edge_detection, self.bilateral_filter, self.blur, self.cartoon, self.emboss] #all filters in list
        self.current_filter = 0 #index

    def no_filter(self, img): #raw
        return img

    def grayscale(self, img): #gray
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def edge_detection(self, img): #converts so that only edges can be seen
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert single channel to BGR
        return edges


    def sepia(self, img): #yellowish throught tint
        img_sepia = np.array(img, dtype=np.float64)
        img_sepia = cv2.transform(img_sepia, np.matrix([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ]))
        img_sepia[np.where(img_sepia > 255)] = 255  # Cap values at 255
        return np.array(img_sepia, dtype=np.uint8)

    def negative(self, img): #opposite
        return cv2.bitwise_not(img)
    
    def emboss(self, img): #dark hdr
        kernel = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])
        img_emboss = cv2.filter2D(img, -1, kernel)
        return cv2.addWeighted(img_emboss, 0.7, img, 0.3, 0)

    def sharpen(self, img): #hdr
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    
    def blur(self, img): #blurred through openCV filter
        return cv2.GaussianBlur(img, (15, 15), 0)

    def bilateral_filter(self, img):
        return cv2.bilateralFilter(img, 9, 75, 75)

    def cartoon(self, img): #edges defined and colored in 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(img, 9, 75, 75)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon


    def apply_filter(self, img):#takes image and filter and applies at the image for all frames

        return self.filters[self.current_filter](img)

    def switch_filter(self): #basic increment and loop
        self.current_filter = (self.current_filter + 1) % len(self.filters)

    def capture_image(self, img): #function to take picture and save
        directory = r'captures'
        if not os.path.exists(directory):
            os.makedirs(directory)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(os.path.join(directory, f'{timestamp}.jpg'), img)


    def run(self):
        cap = cv2.VideoCapture(0) #use webcam default
        pTime = 0 #previous time

        while True:
            success, img = cap.read() #reads images coming in from webcam (all frames)
            if not success: 
                break

            img = self.detector.findHands(img) #use HandTrackingModule class's function
            lmList = self.detector.findPosition(img) #find position of the hands

            filtered_img = self.apply_filter(img)  # Apply filter to the image

            if lmList:
                #Hand signal: Thumbs up to switch filter
                thumb_tip = lmList[4][2] 
                index_tip = lmList[8][2]
                middle_tip = lmList[12][2]

                if thumb_tip < index_tip < middle_tip: #means thumb is above index and middle tip meaning giving thumbs up sign
                    self.switch_filter()

                #Hand signal: Open palm to take picture
                palm_open = all(lmList[i][2] < lmList[i - 2][2] for i in range(8, 21, 4)) #all landmarks can be seen and the fingers are higher than their knuckles
                if palm_open:
                    self.capture_image(filtered_img)

            img = self.apply_filter(img)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime #previous time changed

            cv2.putText(filtered_img, f'FPS: {int(fps)}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
            cv2.putText(filtered_img, f'FILTER: {int(self.current_filter)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            cv2.imshow("Image", filtered_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): #quit key is q
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    FilterControl().run()

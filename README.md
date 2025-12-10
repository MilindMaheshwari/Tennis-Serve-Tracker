# Tennis Serve Tracker

(Progress will be uploaded publicly soon)

 Hello! I am currently developing an app that can provide live feedback on Tennis Serving form. For the first stages of this, I will be working on accurately mapping the trajectory of the ball. The steps for development are as follows: 

 1. Training YOLOv11n model to track tennis balls - based off AlexA's dataset here: https://universe.roboflow.com/alexa-wpmns/tennis-ball-obj-det/. 
         - I will be comparing the performance of this vs. simpler methods of tracking
 2. Mapping trajectory of ball to provide feedback on toss (comparing to ideal trajectory for First Serve tosses)
 3. Using OpenCV OpenPose to track the movement of limbs and racket to watch for errors while swinging

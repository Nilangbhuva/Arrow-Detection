import cv2 
import time

cap = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sample.mp4', fourcc, 20.0, (640, 480))

start_time = time.time()

print("Recording started...")

while True:
    ret, frame = cap.read() 

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    out.write(frame)

    cv2.imshow('Recording', frame)

    if time.time() - start_time > 5:
        print("Recording finished.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Recording interrupted by user.")
        break


cap.release()
out.release()


cv2.destroyAllWindows()

print("Video saved as sample.mp4")


def FrameCapture(path): 


	vidObj = cv2.VideoCapture(path) 


	count = 0


	success = 1

	while success: 


		success, image = vidObj.read() 
		if not success:
			break
		cv2.imwrite("frame%d.jpg" % count, image) 

		count += 1


FrameCapture("sample.mp4") 

image_paths=[] 
for i in range(7,147,8):
    s = 'Path According to Images Saved/frame' + str(i) + '.jpg'
    image_paths.append(s)

imgs = [] 

for i in range(len(image_paths)): 
	imgs.append(cv2.imread(image_paths[i]))  
	# this is optional if your input images isn't too large 
	# you don't need to scale down the image 
	# in my case the input images are of dimensions 3000x1200 
	# and due to this the resultant image won't fit the screen 
	# scaling down the images
 

stitchy=cv2.Stitcher.create() 
(dummy,output)=stitchy.stitch(imgs) 

if dummy != cv2.STITCHER_OK: 
	print("stitching ain't successful") 
else: 
	print('Your Panorama is ready!!!') 
cv2.imwrite("Final.jpg", output) 

cv2.imshow('final result',output) 

cv2.waitKey(0)

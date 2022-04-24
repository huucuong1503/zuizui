# Import socket module
import socket
import time

import cv2
import numpy as np
import math
import timeit
from keras.models import load_model
from PIL import Image, ImageOps

global sendBack_angle, sendBack_Speed, current_speed, current_angle

sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
lm=True
lastState=0;
preangle=0
bias=0
mid=0
speed=40
h=0
w=0
dem=0
check=False
uncheck=False
model = load_model('/root/bainop/keras_model.h5')
data1 = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#
end=0
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('host.docker.internal', PORT))
def chon(a1,a2,b1,b2):
    global mid
    global bias
    midArr=[int((a1+a2)/2)+bias,int((a1+b1)/2)+bias,int((a1+b2)/2)+bias,int((a2+b1)/2)+bias,int((a2+b2)/2)+bias,int((b1+b2)/2)+bias]
    dist=[abs(a1-a2),abs(a1-b1),abs(a1-b2),abs(a2-b1),abs(a2-b2),abs(b1-b2)]
    #print(dist.index(max(dist)))
    mid=midArr[dist.index(max(dist))]

def pid(angles):
    global preangle
    kp=0.5
    kd=0.3
    p_val=kp*angles
    d_val=kd*(angles-preangle)
    return p_val+d_val
def getContours(img):
    global x,y,w,h
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>10:
            cv2.drawContours(img4, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            #print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)


            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"



            cv2.rectangle(img4,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img4,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)

def empty(a):
    pass
def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed


start=timeit.timeit()
if __name__ == "__main__":
    try:
        while True:



            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_spaeed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe

            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)

                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(100)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                    ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)

            try:

                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                        ), -1
                    )


                sam=start-end
                end=time.perf_counter()
                kt=image
                k = image

                #cv2.imwrite('C:/Users/Admin/PycharmProjects/pythonProject27/Resources/bienbao3.png',k)
                #..................................
                '''width, height = 400, 250
                pts1 = np.float32([[240, 158], [398, 158], [0, 258], [640, 258]])
                pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgOutput = cv2.warpPerspective(kt, matrix, (width, height))
                '''#.....................................

                image= image[150:300,:,:]
                blur = cv2.GaussianBlur(image, (5, 5), 0)
                imgHSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                blur2 = cv2.GaussianBlur(k, (5, 5), 0)
                imgHSV2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
                #imgOutput=cv2.cvtColor(imgOutput, cv2.COLOR_BGR2HSV)

                #print(h_min, h_max, s_min, s_max, v_min, v_max)
                lower = np.array([0, 0,225])
                upper = np.array([216, 20, 255])
                #lower2 = np.array([105, 69, 120])
                #upper2 = np.array([114, 255, 190])
                lower2 = np.array([106, 68, 125])
                upper2 = np.array([113, 90, 172])
                lower3 = np.array([18, 129, 45])
                upper3 = np.array([30, 220, 255])
                lower4 = np.array([98, 152, 119])
                upper4 = np.array([110, 228, 182])
                lower5 = np.array([137, 39, 83])
                upper5 = np.array([179, 255, 117])
                mask = cv2.inRange(imgHSV, lower, upper)
                mask2 = cv2.inRange(imgHSV, lower2, upper2)
                mask3 = cv2.inRange(imgHSV, lower3, upper3)
                mask4=cv2.inRange(imgHSV2, lower4, upper4)
                mask5=cv2.inRange(imgHSV2, lower5, upper5)
                #imgOutput=cv2.inRange(imgOutput, lower, upper)
                imgResult = cv2.bitwise_and(image, image, mask=mask)
                arr = []
                arr2=[]
                arr3=[]
                img=cv2.Canny(mask, 180,255)
                img2=cv2.Canny(mask2,180,255)
                img3=cv2.Canny(mask3,180,255)
                img4=cv2.Canny(mask4,180,255)
                img5=cv2.Canny(mask5,180,255)
                #imgOutput=cv2.Canny(imgOutput,180,255)

                lineRow=img[50,:]
                lineRow2=img2[50,:]
                lineRow3=img3[50,:]
                #lineRow3=imgOutput[20,:]

                #print(lineRow)
                #print(lineRow.shape())
                for x,y in enumerate(lineRow):
                    if y==255:
                        arr.append(x)
                for x,y in enumerate(lineRow2):
                    if y==255:
                        arr2.append(x)
                for x,y in enumerate(lineRow3):
                    if y==255:
                        arr3.append(x)
                try:
                    arrmax=max(arr)
                except:
                    arrmax=-1
                try:
                    arrmin=min(arr)
                except:
                    arrmin=-1
                try:
                    arrmax3=max(arr3)
                except:
                    arrmax3=-1
                try:
                    arrmin3=min(arr3)
                except:
                    arrmin3=-1
                distance=arrmax-arrmin
                #print(distance)
                if distance<300:
                    try:
                        arrmax1=max(arr2)
                        arrmin1=min(arr2)
                        #print(arrmin1)
                        print(arrmin1, arrmax1)
                        if arrmin<0:
                            arrmin=arrmin1
                        if arrmax<0:
                            arrmax=arrmax1
                        print("ahihi")
                        chon(arrmax,arrmin,arrmax1,arrmin1)

                        angle = math.degrees(math.atan((mid - img.shape[1] / 2) / (img.shape[0] - 50)))
                        cv2.circle(img, (arrmax1, 50), 5, (255, 255, 255), 5)
                        cv2.circle(img, (arrmin1, 50), 5, (255, 255, 255), 5)
                        cv2.circle(img, (arrmax3, 50), 5, (255, 255, 255), 5)
                        cv2.circle(img, (arrmin3, 50), 5, (255, 255, 255), 5)
                    except:

                        if uncheck==True:
                            if list(prediction[0]).index(max(prediction[0])) == 1:

                                while  abs(tam-end)<2:
                                    print(end - tam)
                                    Control(25,0)
                                    end=time.perf_counter()
                                print(end - tam)
                                tam=end
                                while abs(tam - end) < 3.5:
                                    Control(-25,0)

                                    end=time.perf_counter()
                                print('uhuhu')

                else:

                    mid=int((arrmax+arrmin)/2)+bias

                    uncheck=False
                #mid3=int(arrmax3)

                #angle=math.degrees(math.atan((mid-img.shape[1]/2)/(img.shape[0]-50)))
                #angle3 = math.degrees(math.atan((mid3 - imgOutput.shape[1] / 2) / (imgOutput.shape[0] - 20)))
                angle = math.degrees(math.atan((mid - img.shape[1] / 2) / (img.shape[0] - 50)))

                distance=arrmax-arrmin
                if 255 in img4:
                    getContours(img4)
                    cv2.imshow("res3", k[y:y + h, x:x + w, :])
                if 255 in img5:
                    getContours(img5)
                    cv2.imshow("res4", k[y:y + h, x:x + w, :])

                if dem <=7:
                    dem+=1
                    image1 = Image.open('bienbao.png')
                    # resize the image to a 224x224 with the same strategy as in TM2:
                    # resizing the image to be at least 224x224 and then cropping from the center
                    size = (224, 224)

                    image1 = ImageOps.fit(image1, size, Image.ANTIALIAS)

                    # turn the image into a numpy array
                    image_array = np.asarray(image1)
                    # Normalize the image
                    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                    # Load the image into the array
                    data1[0] = normalized_image_array
                    # run the inference
                    # run the inference
                    prediction = model.predict(data1)
                if 60>=h>=40 and (255 in img4 or 255 in img5):
                    if abs(w-h)<=5:
                        print(h)
                        try:
                            cv2.imwrite('/root/bainop/bienbao.png', k[y:y + h, x:x + w, :])
                            image1 = Image.open('/root/bainop/bienbao.png')
                            # resize the image to a 224x224 with the same strategy as in TM2:
                            # resizing the image to be at least 224x224 and then cropping from the center
                            size = (224, 224)

                            image1 = ImageOps.fit(image1, size, Image.ANTIALIAS)

                            # turn the image into a numpy array
                            image_array = np.asarray(image1)
                            # Normalize the image
                            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                            # Load the image into the array
                            data1[0] = normalized_image_array
                            # run the inference
                            prediction = model.predict(data1)
                            print(list(prediction[0]).index(max(prediction[0])))
                            print(list(prediction[0]).index(max(prediction[0])))
                            if list(prediction[0]).index(max(prediction[0]))==1:
                                check=True

                        except:
                            pass
                if check==True:
                    angle = math.degrees(math.atan((arrmin- img.shape[1] / 2) / (img.shape[0] - 50)))
                    if (255 not in img4 and 255 not in img5):
                        check=False
                        uncheck=True
                        tam=end
                print(arrmin, arrmax)
                if uncheck==False:
                    Control(pid(angle), speed)

                lastState=pid(angle)
                print(lastState)
                #print(gray.shape)
                # your process here


                cv2.waitKey(1)
                #Control(angle, speed)
                #print(distance)
                cv2.circle(img,(arrmin,50),5,(255,255,255),5)
                cv2.circle(img,(arrmax,50),5,(255,255,255),5)
                if (arrmin3!=-1 and arrmax3!=-1):
                    mid = int((arrmax + arrmin) / 2) + bias
                    cv2.circle(img, (arrmax3, 50), 5, (255, 255, 255), 5)
                    cv2.circle(img, (arrmin3, 50), 5, (255, 255, 255), 5)
                    speed=100
                    if abs(arrmin3 - img.shape[1] / 2)>abs(arrmax3 - img.shape[1] / 2):
                        bias=40

                    else:
                        bias=-40
                else:
                    bias=0
                    if uncheck==False:

                        speed=40


                cv2.line(img,(mid,50),(int(img.shape[1]/2),img.shape[0]),(255,255,255),(5))

                #cv2.circle(imgOutput,(arrmin3,20),5,(255,255,255),5)
                #cv2.circle(imgOutput,(arrmax3,20),5,(255,255,255),5)
                #cv2.line(imgOutput,(mid3,20),(int(imgOutput.shape[1]/2),imgOutput.shape[0]),(255,255,255),(5))

                preangle=angle
                #cv2.imshow("res",img)
                #cv2.imshow("res2",img5)
                #cv2.imshow("bu",image)
                #cv2.imshow("12",mask2)
            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()

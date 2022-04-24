# Import socket module
import socket
import cv2
import numpy as np

global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
lastState=0;
preangle=0
bias=0
mid=0
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
    print(dist.index(max(dist)))
    mid=midArr[dist.index(max(dist))]

def pid(angles):
    global preangle
    kp=0.5
    kd=0.3
    p_val=kp*angles
    d_val=kd*(angles-preangle)
    return p_val+d_val

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 216, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 216, 216, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 20, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 225, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed


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

                start=timeit.timeit()
                sam=start-end
                end=start
                kt=image
                cv2.imwrite('C:/Users/Admin/PycharmProjects/pythonProject27/Resources/chuongngai.jpg',kt)
                #..................................
                '''width, height = 400, 250
                pts1 = np.float32([[240, 158], [398, 158], [0, 258], [640, 258]])
                pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgOutput = cv2.warpPerspective(kt, matrix, (width, height))
                '''#.....................................
                image= image[150:300,:,:]
                blur = cv2.GaussianBlur(image, (5, 5), 1)
                imgHSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                #imgOutput=cv2.cvtColor(imgOutput, cv2.COLOR_BGR2HSV)

                h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
                h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
                s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
                s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
                v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
                v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
                #print(h_min, h_max, s_min, s_max, v_min, v_max)
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])
                lower2 = np.array([105, 69, 114])
                upper2 = np.array([114, 255, 255])
                mask = cv2.inRange(imgHSV, lower, upper)
                mask2 = cv2.inRange(imgHSV, lower2, upper2)
                #imgOutput=cv2.inRange(imgOutput, lower, upper)
                imgResult = cv2.bitwise_and(image, image, mask=mask)



                arr = []
                arr2=[]
                arr3=[]
                img=cv2.Canny(mask, 180,255)
                img2=cv2.Canny(mask2,180,255)

                #imgOutput=cv2.Canny(imgOutput,180,255)

                lineRow=img[50,:]
                lineRow2=img2[50,:]
                #lineRow3=imgOutput[20,:]

                #print(lineRow)
                #print(lineRow.shape())
                for x,y in enumerate(lineRow):
                    if y==255:
                        arr.append(x)
                for x,y in enumerate(lineRow2):
                    if y==255:
                        arr2.append(x)
                '''for x,y in enumerate(lineRow3):
                    if y==255:
                        arr3.append(x)'''
                try:
                    arrmax=max(arr)
                except:
                    arrmax=-1
                try:
                    arrmin=min(arr)
                except:
                    arrmin=-1
                    #arrmax3=max(arr3)
                    #arrmin3=min(arr3)
                distance=arrmax-arrmin
                print(distance)
                if distance<280:
                    try:
                        arrmax1=max(arr2)
                        arrmin1=min(arr2)
                        if arrmin<0:
                            arrmin=arrmin1
                        if arrmax<0:
                            arrmax=arrmax1
                        chon(arrmax,arrmin,arrmax1,arrmin1)
                        cv2.circle(img, (arrmax1, 50), 5, (255, 255, 255), 5)
                        cv2.circle(img, (arrmin1, 50), 5, (255, 255, 255), 5)
                    except:
                        pass
                else:
                    mid=int((arrmax+arrmin)/2)+bias
                #mid3=int(arrmax3)

                angle=math.degrees(math.atan((mid-img.shape[1]/2)/(img.shape[0]-50)))
                #angle3 = math.degrees(math.atan((mid3 - imgOutput.shape[1] / 2) / (imgOutput.shape[0] - 20)))

                distance=arrmax-arrmin
                Control(pid(angle), 40)
                lastState=pid(angle)
                #print(gray.shape)
                # your process here


                cv2.waitKey(1)
                #Control(angle, speed)
                #print(distance)

                cv2.circle(img,(arrmin,50),5,(255,255,255),5)
                cv2.circle(img,(arrmax,50),5,(255,255,255),5)
                cv2.line(img,(mid,50),(int(img.shape[1]/2),img.shape[0]),(255,255,255),(5))

                #cv2.circle(imgOutput,(arrmin3,20),5,(255,255,255),5)
                #cv2.circle(imgOutput,(arrmax3,20),5,(255,255,255),5)
                #cv2.line(imgOutput,(mid3,20),(int(imgOutput.shape[1]/2),imgOutput.shape[0]),(255,255,255),(5))

                preangle=angle
                cv2.imshow("res",img)
                cv2.imshow("res2",img2)
                #cv2.imshow("bu",image)
                #cv2.imshow("12",mask2)
            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()

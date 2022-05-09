'''
    File name: VisualAssistant.py
    Author: Michele Pompilio
    Date created: 01/08/2021
    Date last modified: 14/09/2021
    Python Version: 3.9
'''
import os
import cv2
import numpy as np
import time
import webbrowser
import cvlib
from cvlib.object_detection import draw_bbox

traverse_point = []
traverse_point_s = []
options = ['Search', 'Zoom in', 'Zoom out', 'Exit']

def read_menu_images(path):
    filespath = os.listdir(path)
    menu_images = []
    for filepath in filespath:
        img = cv2.imread(path+'/'+filepath)
        if img is None:
            menu_images.clear()
            menu_images = None
            print("Can't load menu images. Menu won't be shown.")
            break
        menu_images.append(img)
    return menu_images




def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def check_intersection(pointer, bbox):
    if pointer is not None and bbox is not None:
        for i in range(len(bbox)):
            if  bbox[i][0]<= pointer[0] <= bbox[i][2] and bbox[i][1] <= pointer[1] <= bbox[i][3] :
                return True, i
    return False, None

def contours(mask_image):
    ret, thresh = cv2.threshold(mask_image, 200, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont

def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=float)
        y = np.array(contour[s][:, 0][:, 1], dtype=float)

        pos = np.where(y <= cy)
        x = x[pos]
        y = y[pos]

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(y):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, scene_f, fx, fy, mask_image):
    mask_image = cv2.erode(mask_image, None, iterations=2)
    mask_image = cv2.dilate(mask_image, None, iterations=2)
    cv2.imshow("debug", mask_image)

    contour_list = contours(mask_image)
    max_cont = max(contour_list, key=cv2.contourArea)

    cnt_centroid = centroid(max_cont)
    #cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)
    #cv2.drawContours(frame, [max_cont], 0, (255,0,255), 3)
    far_point_s=None
    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        far_point_s = (int(far_point[0] * fx), int(far_point[1] * fy))

        #show cvx hull
        '''for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_cont[s][0])
            end = tuple(max_cont[e][0])
            cv2.line(frame, start, end, [255, 0, 255], 2)
            cv2.circle(frame, start, 4, [255, 0, 0], -1)'''

        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        cv2.circle(scene_f, far_point_s, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
            traverse_point_s.append(far_point_s)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)
            traverse_point_s.pop(0)
            traverse_point_s.append(far_point_s)

        draw_circles(frame, traverse_point)
        draw_circles(scene_f, traverse_point_s)

    return far_point_s

def gesture_recognition(frame, mask_image):
    mask_image = cv2.erode(mask_image, None, iterations=2)
    mask_image = cv2.dilate(mask_image, None, iterations=2)
    cv2.imshow("debug", mask_image)

    contour_list = contours(mask_image)
    cnt = max(contour_list, key=cv2.contourArea)

    # l = no. of defects
    l = 0
    gesture = None
    if cnt is not None:
        cx, cy = centroid(cnt)
        cv2.circle(frame, (cx, cy), 5, [255, 0, 255], -1)

        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(frame, [approx], 0, (255, 0, 255), 2)

        # make convex hull around hand
        hull = cv2.convexHull(cnt)

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

        # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        if defects is not None:
            # code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                # find length of all sides of triangle
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = np.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # apply cosine rule here
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= np.pi and d > 40:
                    l += 1
                    cv2.circle(frame, far, 3, [255, 0, 0], -1)

                # draw lines around hand
                cv2.line(frame, start, end, [0, 255, 0], 2)

            l += 1

            if l == 1:
                if arearatio < 12:
                    gesture = 0
                elif arearatio > 17.5:
                    gesture = 1

            elif l == 2:
                gesture = 2

            elif l == 3:
                if arearatio < 27:
                    gesture = 3

            elif l == 4:
                gesture = 4

            elif l == 5:
                gesture =5

    return gesture

def zoom_obj(scene_f, bbox, label, m=2):
    obj = scene_f[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if obj.shape[0] * m < scene_f.shape[0] and obj.shape[1] * m < scene_f.shape[1]:
        obj = cv2.resize(obj, (int(obj.shape[1] * m), int(obj.shape[0] * m)), interpolation=cv2.INTER_AREA)
        x1_pos, x2_pos = int(scene_f.shape[1] / 2 - obj.shape[1] / 2), int(scene_f.shape[1] / 2 + obj.shape[1] / 2)
        y1_pos, y2_pos = int(scene_f.shape[0] / 2 - obj.shape[0] / 2), int(scene_f.shape[0] / 2 + obj.shape[0] / 2)
        scene_f[y1_pos:y2_pos, x1_pos:x2_pos] = obj
        cv2.putText(scene_f, label+" X"+str(m), (0, scene_f.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Scene", scene_f)

def draw_menu(frame, menu_images, x_pos, y_pos):
    global options
    h, w = frame.shape[:2]
    m_h, m_w = int(0.8 * h), int(0.48 * w)
    img_size = int(m_h/4)
    window = np.zeros((m_h,m_w,3))
    cv2.rectangle(window,(0,0),(m_w,m_h),(255,255,255),-1)
    for ix, img in enumerate(menu_images):
        icon = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_AREA)
        window[ix*img_size:ix*img_size+img_size, :img_size] = icon
        cv2.line(window, (0, ix*img_size), (m_w, ix*img_size), (255, 0, 0), 2)
        cv2.putText(window, options[ix], (img_size+10, ix*img_size+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(window, (0, 0), (m_w, m_h), (255, 0, 0), 2)
    cv2.line(window,(img_size,0),(img_size,m_h),(255,0,0),2)
    frame[y_pos:y_pos+m_h, x_pos:x_pos+m_w] = window

def highlights_option(frame, idx, x_pos, y_pos):
    global options
    if idx >= 2:
        h, w = frame.shape[:2]
        m_h, m_w = int(0.8 * h), int(0.48 * w)
        img_size = int(m_h / 4)
        cv2.rectangle(frame,(x_pos+img_size, y_pos+idx*img_size), (x_pos+m_w, y_pos+idx*img_size+img_size), (0, 255, 0), -1)
        cv2.rectangle(frame,(x_pos+img_size, y_pos+idx*img_size), (x_pos+m_w, y_pos+idx*img_size+img_size), (255, 0, 0), 2)
        cv2.putText(frame, options[idx], (x_pos+img_size+10, y_pos+idx*img_size+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def main():

    brows = webbrowser.get('windows-default')
    capture = cv2.VideoCapture(0)
    scene = cv2.VideoCapture('./your/video/path')
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    menu_images = read_menu_images("./menu_img")

    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Scene", cv2.WINDOW_NORMAL)

    frame_rate = 30
    waiting_det = 0
    waiting_init = 0
    waiting_inter = 0
    waiting_comm = 0

    prev = time.time()

    bbox = None
    label = None
    idx = None
    conf = None
    or_scene = None
    fgMask = None
    command_mode = False
    init_done = False
    zoomed = False
    prev_comm = 0
    while capture.isOpened():
        if not scene.isOpened():
            break


        time_elapsed = time.time() - prev
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        ret, read_scene = scene.read()

        if pressed_key & 0xFF == ord('i'):
            backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            waiting_init = 0
            init_done = False

        if not command_mode:
            scene_f = read_scene
            or_scene = np.copy(scene_f)

        if time_elapsed > 1./frame_rate:
            waiting_det += time_elapsed
            waiting_init += time_elapsed
            prev = time.time()

            if ret is True:
                h, w = frame.shape[:2]
                h_s, w_s = scene_f.shape[:2]

                fx, fy = w_s / (0.35 * w), h_s / (0.4 * h)
                frame = cv2.flip(frame, 1)

                pointer = None

                cv2.rectangle(frame,(int(0.5*w),0),(w,int(0.8*h)),(0,255,0),2)
                roi = frame[:int(0.8 * h), int(0.5 * w):]

                if waiting_init < 3:
                    fgMask = backSub.apply(roi)
                    cv2.rectangle(frame, (int(w / 4), int(h * 3 / 8)+30), (int(w * 3 / 4)+50, int(h * 5 / 8)-40), (255, 255, 255),-1)
                    cv2.putText(frame, 'initialization, stay out from the marked area...', (int(w / 4), int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 1, cv2.LINE_AA)
                else:
                    fgMask = backSub.apply(roi, learningRate=0)
                    if init_done is False:
                        init_done = True

                if init_done is True:
                    if command_mode is False:
                        #finger tracking
                        try:
                            pointer = manage_image_opr(roi, scene_f, fx, fy, fgMask)
                        except:
                            cv2.rectangle(frame,(int(w/4),int(h*3/8)),(int(w*3/4),int(h*5/8)),(0,0,255),-1)
                            cv2.putText(frame, 'No hand detected', (int(w/4)+20,int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

                        if prev == 0 or waiting_det > 1:
                            waiting_det = 0
                            bbox, label, conf = cvlib.detect_common_objects(scene_f, confidence=0.4, model='yolov4-tiny')
                        scene_f = draw_bbox(scene_f, bbox, label, conf, write_conf=True)

                        ret, idx = check_intersection(pointer, bbox)
                        if ret:
                            waiting_inter += time_elapsed
                        else:
                            waiting_inter = 0
                        if waiting_inter > 1:
                            waiting_inter = 0
                            ###command mode
                            scene_f = draw_bbox(np.copy(or_scene), [bbox[idx]], [label[idx]], [conf[idx]], write_conf=True)
                            command_mode = True
                            prev_comm = 0
                        cv2.imshow("Scene", scene_f)
                    else:
                        #gesture recognition
                        if menu_images is not None:
                            draw_menu(frame, menu_images, 0, 0)
                        try:
                            cnt = gesture_recognition(roi,fgMask)
                            cv2.putText(roi, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                            if cnt == prev_comm:
                                waiting_comm += time_elapsed
                                if menu_images is not None:
                                    highlights_option(frame, cnt-2, 0, 0)
                            else:
                                prev_comm = cnt
                                waiting_comm = 0

                            if waiting_comm > 1:
                                waiting_comm = 0
                                if cnt == 2:
                                    brows.open('www.wikipedia.org/wiki/' + label[idx])
                                elif cnt == 3 and zoomed is False:
                                    zoom_obj(scene_f, bbox[idx], label[idx])
                                    zoomed = True
                                elif cnt == 4 and zoomed is True:
                                    scene_f = draw_bbox(np.copy(or_scene), [bbox[idx]], [label[idx]], [conf[idx]], write_conf=True)
                                    cv2.imshow("Scene", scene_f)
                                    zoomed = False
                                elif cnt == 5:
                                    command_mode = False
                                    if zoomed is True:
                                        zoomed = False


                        except:
                            cv2.rectangle(frame, (int(w / 4), int(h * 3 / 8)), (int(w * 3 / 4), int(h * 5 / 8)),(0, 0, 255), -1)
                            cv2.putText(frame, 'No gesture detected', (int(w / 4), int(h / 2)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

                    frame[:int(0.8 * h), int(0.5 * w):] = roi
                    cv2.line(frame, (int(0.5 * w), int(0.4 * h)), (int(0.85 * w), int(0.4 * h)), (0, 0, 255), 1)
                    cv2.line(frame, (int(0.85 * w), 0), (int(0.85 * w), int(0.4 * h)), (0, 0, 255), 1)
                cv2.imshow("Live Feed", rescale_frame(frame))

                if pressed_key == 27:
                    break
            else:
                scene.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()

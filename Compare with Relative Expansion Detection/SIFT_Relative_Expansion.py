# coding=utf-8
import os,time
import numpy as np
# from PIL import Image,ImageGrab
from matplotlib import pyplot as plt
import cv2
import xlrd,xlwt


Frame_start = 1
Frame_End = 62
List_of_Scales = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
TM_Scale_Res = [0, 0, 0, 0, 0, 0, 0, 0]
Score_of_Scales = [0, 1, 1, 2, 2, 2, 3, 3]      #Give the matched scale a score, to transfer recognised obstacle into Membrane Potential.



def from_kp_to_roi(x, y, size,img_width,img_height):
    roi_x1 = int(x) - int(size)
    roi_x2 = int(x) + int(size)
    roi_y1 = int(y) - int(size)
    roi_y2 = int(y) + int(size)

    if roi_x1 < 0:
        roi_x1 = 0
    if roi_x2 > img_width:
        roi_x2 = img_width
    if roi_y1 < 0:
        roi_y1 = 0
    if roi_y2 > img_height:
        roi_y2 = img_height
    rect_roi = [roi_x1,roi_x2,roi_y1,roi_y2]
    return rect_roi


SIFT = cv2.SIFT_create()
bf_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)

Ratio_kp = []
Ratio_Hull = []
Ratio_TM = []
Obstacle_Scales = []
#Go through the video frames:
for k in range(Frame_start,Frame_End):
    GoodMatches = []
    Expanding_KP1 = []
    Expanding_KP2 = []
    Expanding_KP1_After_TM = []
    Expanding_KP2_After_TM = []
    img_1 = cv2.imread('Figures/Chair1/DemoFlight_{}.bmp'.format(str(k)))
    img_2 = cv2.imread('Figures/Chair1/DemoFlight_{}.bmp'.format(str(k+1)))
    Obstacle_count = 0
    height, width = img_1.shape[:2]

    Origin_Grey1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    Origin_Grey2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    kp1, query_des = SIFT.detectAndCompute(Origin_Grey1, None)
    kp2, train_des = SIFT.detectAndCompute(Origin_Grey2, None)
    matches = bf_matcher.knnMatch(query_des, train_des, 2)
    print('Frame_' + str(k) + ': Origen matches:' + str(len(matches)))
    for m, n in matches:  # Select out good matches     m,n for the best and the second match respectively.
        if m.distance < 0.5 * n.distance:
            if m.distance < 400:        #400 for simulation, 100 for real scene
                if kp2[m.trainIdx].size > kp1[m.queryIdx].size:  # Ensuring the size of the keypoint is expanding
                    GoodMatches.append(m)
                    Expanding_KP1.append(kp1[m.queryIdx])
                    Expanding_KP2.append(kp2[m.trainIdx])
    print('Selected good matches:' + str(len(GoodMatches)))
    GoodMatches = sorted(GoodMatches, key=lambda x: x.distance)
    # PointSets_1 = cv2.KeyPoint_convert(Expanding_KP1)  # acquire (x,y) from the keypoints.
    # PointSets_2 = cv2.KeyPoint_convert(Expanding_KP2)  # acquire (x,y) from the keypoints.

    Scales = 0
    for i in GoodMatches:  # Template size ratio
        Minimal_Scale_Diff = 1000000
        size_origin = kp1[i.queryIdx].size
        for x in range(0, 8):  # Find the scale with the minimal difference
            scale = List_of_Scales[x]
            Rect_ROI1 = from_kp_to_roi(kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1], size_origin, width, height)
            Rect_ROI2 = from_kp_to_roi(kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1], size_origin * scale, width, height)
            # cv2.rectangle(img_1, (Rect_ROI1[0], Rect_ROI1[2]), (Rect_ROI1[1], Rect_ROI1[3]), (0, 255, 0),3)
            try:  # if the point nears the broundary of the image, BUG may occur
                Template1_origin = Origin_Grey1[Rect_ROI1[2]:Rect_ROI1[3], Rect_ROI1[0]:Rect_ROI1[1]]
                Template2 = Origin_Grey2[Rect_ROI2[2]:Rect_ROI2[3],
                            Rect_ROI2[0]:Rect_ROI2[1]]  # cut the template (current t) around the keypoints.
                Template1_scaled = cv2.resize(Template1_origin, (
                    Template2.shape[1], Template2.shape[0]))  # resize template1 to the same scale as template2
            except:
                print('Size Error, Cannot acquire the template')
            Res_TM = cv2.absdiff(Template1_scaled, Template2)
            S = Res_TM.sum()
            S_Relative = S / (scale * scale)
            TM_Scale_Res[x] = S_Relative
            if (S_Relative < Minimal_Scale_Diff) & (S_Relative != 0):  # select the best (minimal) matched scale
                Minimal_Scale_Diff = S_Relative
                Minimal_Scale_Idx = x
            if x == 0:
                TM_scale_0_Res = S_Relative
        if (Minimal_Scale_Diff < 0.8 * TM_scale_0_Res) & (
                List_of_Scales[Minimal_Scale_Idx] > 1.2):  # Select the Expanding Templates
            Obstacle_count = Obstacle_count + 1
            print('In Frame_%d, Obstacle_%d identified,S = %d, Best_Scale = %f' % (
            k, Obstacle_count, S, List_of_Scales[Minimal_Scale_Idx]))
            # cv2.rectangle(img_1, (Rect_ROI1[0], Rect_ROI1[2]), (Rect_ROI1[1], Rect_ROI1[3]), (0, 255, 0),
            #               3)  # Draw rectangles representing for expanding templates
            Obstacle_Scales.append(List_of_Scales[Minimal_Scale_Idx])
            # Expanding_KP1_After_TM.append(kp1[i.queryIdx])
            # Expanding_KP2_After_TM.append(kp1[i.trainIdx])
        else:
            GoodMatches.remove(i)  # Remove the points of non-expanding templates

    PointSets_1 = cv2.KeyPoint_convert(Expanding_KP1)  # acquire (x,y) from the keypoints.
    PointSets_2 = cv2.KeyPoint_convert(Expanding_KP2)  # acquire (x,y) from the keypoints.
    Size_1=Size_2=0
    for i in GoodMatches:
        Size_1 = Size_1 + kp1[i.queryIdx].size
        Size_2 = Size_2 + kp2[i.trainIdx].size
    if len(GoodMatches)>=3:
        Avg_Size_1 = Size_1/len(GoodMatches)
        Avg_Size_2 = Size_2/len(GoodMatches)
        Ratio_kp.append (Avg_Size_2/Avg_Size_1)
        # Draw the hull.
        hull_1 = cv2.convexHull(PointSets_1)
        length = len(hull_1)
        for q in range(len(hull_1)):
            temp_x = int(hull_1[q][0][0])
            temp_y = int(hull_1[q][0][1])
            temp_x2 = int(hull_1[(q + 1) % length][0][0])
            temp_y2 = int(hull_1[(q + 1) % length][0][1])
            cv2.line(img_1, (temp_x, temp_y), (temp_x2, temp_y2), (0, 255, 0), 2)

        hull_2 = cv2.convexHull(PointSets_2)
        length = len(hull_2)
        for q in range(len(hull_2)):
            temp_x = int(hull_2[q][0][0])
            temp_y = int(hull_2[q][0][1])
            temp_x2 = int(hull_2[(q + 1) % length][0][0])
            temp_y2 = int(hull_2[(q + 1) % length][0][1])
            cv2.line(img_1, (temp_x, temp_y), (temp_x2, temp_y2), (0, 0, 255), 2)
        area1 = cv2.contourArea(hull_1)
        area2 = cv2.contourArea(hull_2)
        if area1 != 0:
            Ratio_Hull.append(area2 / area1)
        else:
            Ratio_Hull.append(1.0)


    else:
        Ratio_kp.append(1.0)
        Ratio_Hull.append(1.0)
        print('No Good matches')



    Avg_scales = 0
    if len(Obstacle_Scales) != 0:
        for j in Obstacle_Scales:
            Avg_scales = Avg_scales+j
        Avg_scales = Avg_scales/len(Obstacle_Scales)
        Ratio_TM.append(Avg_scales)
    else:
        Ratio_TM.append(1)




    cv2.drawKeypoints(img_1, Expanding_KP1, img_1, color=(255, 0, 0),
                      flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # h, w = img_1.shape[:2]
    # Show_Resized = cv2.resize(img_1, dsize=(w // 2, h // 2))
    # cv2.imshow('Expanding_kps', Show_Resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('Output/Detected_Chair_Hull/Chair1_Hull_{}.bmp'.format(str(k)),img_1)

x = np.arange(Frame_start,Frame_End)
plt.plot(x, Ratio_kp, 'green', x, Ratio_Hull, 'red')
plt.show()

# workbook = xlwt.Workbook(encoding="utf-8")
# sheet = workbook.add_sheet('OutputSheet1')
# index_temp = 0
# for j in Ratio_kp:
#     index_temp += 1
#     sheet.write(index_temp, 1, j)
# index_temp = 0
# for j in Ratio_Hull:
#     index_temp += 1
#     sheet.write(index_temp, 2, j)
# workbook.save('Output/Data_Chair.xls')






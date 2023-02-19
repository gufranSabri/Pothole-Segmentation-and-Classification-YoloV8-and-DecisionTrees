from roboflow import Roboflow
import numpy as np
import pandas as pd
import os
import cv2
import sys

def extract_data(zone,rowmin,rowmax,colmin,colmax):
    pothole_area = 0
    for i in range(rowmin+1,rowmax):
        border_encountered_count = 0
        in_border = False
        for j in range(colmin,colmax+1):
            try:
                if list(zone[i,j]) == [255,0,0]:
                    if not in_border: border_encountered_count+=1
                    in_border = True
                    continue
                in_border = False
                if border_encountered_count%2!=0:
                    pothole_area+=1
                    zone[i,j] = np.array([0,255,0])
            except:
                continue

    h = rowmax-rowmin
    w = colmax-colmin
    area = h*w
    non_pothole_area = max(area - pothole_area,0)

    return [h,w,area,pothole_area,non_pothole_area]

def get_box_corners(points):
    pts = []
    
    rowmin,rowmax,colmin,colmax = 1000000,-1,1000000,-1
    for pt in points:
        rowmin = int(min(rowmin,pt["y"]))
        rowmax = int(max(rowmax,pt["y"]))
        colmin = int(min(colmin,pt["x"]))
        colmax = int(max(colmax,pt["x"]))

        pts.append([int(pt["x"]),int(pt["y"])])
    
    return (pts,rowmin,rowmax,colmin,colmax)

img_data_path = '/media/gufran/GsHDD/Work/Projects/AI/PotholeSeverityClassification/Data/SegmentationData/YoloV8Aug'
csv_src_path = '/media/gufran/GsHDD/Work/Projects/AI/PotholeSeverityClassification/Data/ClassificationData'

rf = Roboflow(api_key="fMypfsDsmDRn1Ekv1k82")
segmodel = rf.workspace().project("pothole-segmentation-hqol4").version(2).model

cols = ["imgpath","height","width","area","pothole_area","nonpothole_area"]
df_zone2 = pd.DataFrame(columns=cols)
df_zone3 = pd.DataFrame(columns=cols)

splits = ["train", "valid", "test"]
for s in splits:
    curr_path = img_data_path + '/' + s + '/images'
    images = os.listdir(curr_path)

    index = 0
    print("\nProcessing "+s +" images")
    for im in images:
        print(index, end=" ")
        sys.stdout.flush()
        index+=1
        
        img = cv2.imread(curr_path+"/"+im)
        h,w,_ = img.shape

        zone2 = img[h//4:h//2]
        zone3 = img[h//2:]

        cv2.imwrite("./cache/"+im.replace(".jpg","")+"z2.jpg", zone2)
        cv2.imwrite("./cache/"+im.replace(".jpg","")+"z3.jpg", zone3)

        zone2_pred = segmodel.predict("./cache/"+im.replace(".jpg","")+"z2.jpg").json()
        zone3_pred = segmodel.predict("./cache/"+im.replace(".jpg","")+"z3.jpg").json()

        rowminz2,rowmaxz2,colminz2,colmaxz2=1000000,-1,1000000,-1
        rowminz3,rowmaxz3,colminz3,colmaxz3=1000000,-1,1000000,-1

        for preds in zone2_pred["predictions"]:
            points = preds["points"]

            pts,rowminz2,rowmaxz2,colminz2,colmaxz2 = get_box_corners(points)

            pts = np.array(pts).reshape((-1, 1, 2))
            zone2 = cv2.polylines(zone2, [pts],True, (255, 0, 0), 1)

            data_zone2 = extract_data(zone2,rowminz2,rowmaxz2,colminz2,colmaxz2)
            df_zone2 = pd.concat([df_zone2, pd.DataFrame([[s + '/images' + im]+data_zone2], columns=cols)], axis=0, ignore_index=True)

        for preds in zone3_pred["predictions"]:
            points = preds["points"]

            pts,rowminz3,rowmaxz3,colminz3,colmaxz3 = get_box_corners(points)

            pts = np.array(pts).reshape((-1, 1, 2))
            zone3 = cv2.polylines(zone3, [pts],True, (255, 0, 0), 1)

            data_zone3 = extract_data(zone3,rowminz3,rowmaxz3,colminz3,colmaxz3)
            df_zone3 = pd.concat([df_zone3, pd.DataFrame([[s + '/images' + im]+data_zone3], columns=cols)], axis=0, ignore_index=True)

        os.remove("./cache/"+im.replace(".jpg","")+"z2.jpg")
        os.remove("./cache/"+im.replace(".jpg","")+"z3.jpg")

    print("\n------------------------------------------------------", end='\n')

df_zone2.to_csv(csv_src_path+"/dataz2.csv", index=False)
df_zone3.to_csv(csv_src_path+"/dataz3.csv", index=False)

print(df_zone2.shape)
print(df_zone3.shape)
print(df_zone2.head())
print(df_zone3.head())
import numpy as np
import csv
import cv2
import glob
import os
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
import imutils as im

def HogDiscripter(size,xTimes):
    block_size = (size[0] // (xTimes//2), size[1] // (xTimes//2))
    block_stride = (size[0] // xTimes, size[1] // xTimes)
    cell_size = block_stride
    num_bins = 9
    return cv2.HOGDescriptor(small_size, block_size, block_stride,
                            cell_size, num_bins)

	
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou




SPLIT_RATIO = 0.9
XMLS = "./annotations/xmls"
PNGS = "./images"
class_names = {}
k = 0
output = []
xml_files=""
for i in range(200):
    xml_files=glob.glob("{}/".format(XMLS)+"pomeranian_"+str(i)+".xml")
    if(xml_files):
        tree = ET.parse(xml_files[0])
        path = os.path.join(PNGS, tree.findtext("./filename"))
        basename = os.path.basename(path)
        basename = os.path.splitext(basename)[0]
        class_name = basename[:basename.rfind("_")].lower()

        if class_name not in class_names:
            class_names[class_name] = k
            k += 1
        


        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        xmin = int(tree.findtext("./object/bndbox/xmin"))
        ymin = int(tree.findtext("./object/bndbox/ymin"))
        xmax = int(tree.findtext("./object/bndbox/xmax"))
        ymax = int(tree.findtext("./object/bndbox/ymax"))

        

        output.append((path, height, width, xmin, ymin, xmax, ymax, class_name, class_names[class_name]))

# preserve percentage of samples for each class ("stratified")
output.sort(key=lambda tup : tup[-1])


lengths = []
i = 0
last = 0
j=0
for j, row in enumerate(output):
    if last == row[-1]:
        i += 1
    else:
        print("class {}: {} images".format(output[j-1][-2], i))
        lengths.append(i)
        i = 1
        last += 1

print("class {}: {} images".format(output[j-1][-2], i))
lengths.append(i)

training_data = []
test_data = []
s = 0
for c in lengths:
    for i in range(c):
        path, height, width, xmin, ymin, xmax, ymax, class_name, class_id = output[s]
        if xmin >= xmax or ymin >= ymax or xmax > width or ymax > height or xmin < 0 or ymin < 0:
            print("Warning: {} contains invalid box. Skipped...".format(path))
            continue
            
        if i <= c * SPLIT_RATIO:
            training_data.append(output[s])
        else:
            test_data.append(output[s])

        s += 1
print(len(training_data))
print(len(test_data))
# for b in range(len(training_data)):
#     print (training_data[b])







NewSize=96
xTimes=12



tranDataPathPositive="./tranData/Positive"
tranDataPathNegative="./tranData/Negative"
testDataPath="./testData"

className=training_data[0][-2]
# tranDataNeg=[]

# tranPosImg=[]
# tranNegImg=[]
# testImg=[]
# for i in range (len(training_data)):
#     img=cv2.imread(training_data[i][0],1)
#     rows,cols,ch=img.shape
#     Posimg=img[ training_data[i][4]:training_data[i][6] , training_data[i][3]:training_data[i][5] , :]
#     Posimg=cv2.resize(Posimg,(NewSize,NewSize))
#     tranPosImg.append(Posimg)
#     Negimg=img[ training_data[i][4]:training_data[i][6] ,training_data[i][5]:rows, :]
#     Negrows,Negcols,Nch=Negimg.shape
#     temp=[training_data[i][5],training_data[i][4],rows,training_data[i][6]]
#     if(Negcols<50):
#         Negimg=img[ training_data[i][4]:training_data[i][6] ,0:training_data[i][3], :]
#         temp=[0,training_data[i][4],training_data[i][3],training_data[i][6]]
#     tranDataNeg.append(temp)
#     Negimg=cv2.resize(Negimg,(NewSize,NewSize))
#     tranNegImg.append(Negimg)
# for i in range (len(test_data)):
#     img=cv2.imread(test_data[i][0],1)
#     img=img[ test_data[i][4]:test_data[i][6] , test_data[i][3]:test_data[i][5] , :]
#     img=cv2.resize(img,(NewSize,NewSize))
#     testImg.append(img)


try:
    os.makedirs(tranDataPathPositive)
    os.makedirs(tranDataPathNegative)
    os.mkdir(testDataPath)



    for i in range (len(training_data)):
        img=cv2.imread(training_data[i][0],1)
        rows,cols,ch=img.shape
        Posimg=img[ training_data[i][4]:training_data[i][6] , training_data[i][3]:training_data[i][5] , :]
        Posimg=cv2.resize(Posimg,(NewSize,NewSize))
        cv2.imwrite(tranDataPathPositive+"/"+className+'_'+str(i)+".jpg",Posimg)

        Negimg=img[ training_data[i][4]:training_data[i][6] ,training_data[i][5]:rows, :]
        Negrows,Negcols,Nch=Negimg.shape
        if(Negcols<50):
            Negimg=img[ training_data[i][4]:training_data[i][6] ,0:training_data[i][3], :]
        Negimg=cv2.resize(Negimg,(NewSize,NewSize))
        cv2.imwrite(tranDataPathNegative+"/"+className+'_'+str(i)+".jpg",Negimg)   
    for i in range (len(test_data)):
        img=cv2.imread(test_data[i][0],1)
        cv2.imwrite(testDataPath+"/"+className+'_'+str(i)+".jpg",img)
except:
    print("Done")

# for b in range(len(tranDataNeg)):
#     print (tranDataNeg[b])



small_size = (NewSize, NewSize)

hog = HogDiscripter(small_size,xTimes)




d=9*4*11*11
totalNumberOfTranImages=90*2
HogGrid=np.zeros((totalNumberOfTranImages,d),np.float)
k=0
i=0
tranLables=np.zeros((180),np.uint8)
j=1
while i<90:
    img=cv2.imread(tranDataPathPositive+"/"+className+'_'+str(i)+'.jpg',0)
    HogGrid[k] = hog.compute(img)[:,0]
    k+=1
    img=cv2.imread(tranDataPathNegative+"/"+className+'_'+str(i)+'.jpg',0)
    HogGrid[k] = hog.compute(img)[:,0]
    tranLables[j]=1
    j+=2
    i+=1
    k+=1

# tranData=[]
# for i in range(len(training_data)):
#     tranData.append([training_data[i][3],training_data[i][4],training_data[i][5],training_data[i][5]])
#     tranData.append(tranDataNeg[i])
# print(tranData)
classifier = RandomForestClassifier(n_estimators = 50, max_depth=None,
min_samples_split=2, random_state=0)
classifier.fit(HogGrid,tranLables)

# print (classifier.predict(Hog))

MaxNumOfWinSize=180
size=88
imageNum=3
img=cv2.imread(testDataPath+"/"+className+'_'+str(imageNum)+".jpg",0)
ResultImgs=[]
CropedImgs=[]
Hog=np.zeros((1,d),np.float)
detected="./detected"
try:
    os.makedirs(detected)
except:
    print ("created")
cv2.imwrite(detected+"/"+className+'_real'+'_'+str(imageNum)+'.jpg',img) 
rows,cols=img.shape
print (img.shape)  
i=0  
maxIOU=0.0
print("It'll take 1-3 mins")
while(size<=MaxNumOfWinSize):
    y=0
    while(y+size<rows):
        x=0

        while(x+size<cols):
            NewEximg=img[y:size+y , x:size+x]
            # print(Newimg.shape)
            # cv2.imwrite(detected+"/"+className+'_'+str(i)+'.jpg',Newimg)
            i+=1
            Newimg=cv2.resize(NewEximg,(NewSize,NewSize))
            Hog[0] = hog.compute(Newimg)[0,:]
            Prob=classifier.predict_proba(Hog)
            P=Prob[0][0]
            if(classifier.predict(Hog)[0]==0):
                # print(classifier.predict(Hog)[0])
                # [Newimg,classifier.predict_proba(Hog),x,y,x+size,y+size]
                # if(P>=0.6):
                #     cv2.imwrite(detected+"/"+className+'_'+str(i)+'.jpg',NewEximg)
                #     print(P)
                boxB=[x,y,x+size,y+size]
                boxA=[test_data[imageNum][3],test_data[imageNum][4],test_data[imageNum][5],test_data[imageNum][6]]
                IOU=bb_intersection_over_union(boxA,boxB)
                if(IOU > maxIOU):
                    maxIOU= IOU
                    ResultImgs.append([NewEximg,classifier.predict_proba(Hog),x,y,x+size,y+size])
            x+=4
        y+=4
    size+=4



if(maxIOU!=0.0):
    cv2.imwrite(detected+"/"+className+'_realCroped'+'_'+str(imageNum)+'.jpg',ResultImgs[len(ResultImgs)-1][0])
    cv2.imwrite(detected+"/"+className+'_Predected'+'_'+str(imageNum)+'.jpg',img[test_data[imageNum][4]:test_data[imageNum][6],test_data[imageNum][3]:test_data[imageNum][5]])
# cv2.imshow("result.jpg",ResultImgs[len(ResultImgs)-1][0])

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(ResultImgs)

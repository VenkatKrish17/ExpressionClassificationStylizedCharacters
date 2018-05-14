import os
import cv2

path="./Data/FERG_DB_256/"
output="./preprocessed/"
print(os.listdir(path))
directories=os.scandir(path)
count=1
for content in directories:
    if(content.is_dir()):
        subdir=os.scandir(str(content.path))
        for expr in subdir:
            if(expr.is_dir()):
                print(expr.path)
                img_content=os.scandir(str(expr.path))
                expression=str(expr.path).split("_")[-1]
                output_dir=output+expression
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for i in img_content:
                    filename=output_dir+"/"+expression+"_"+str(count)+".jpg"
                    count+=1
                    print(filename)
                    img=cv2.imread(str(i.path),255)
                    cv2.imwrite(filename,img)
                    #cv2.imshow(str(expr.path),img)
                    #cv2.imwrite(output+expr+'/'+)
                #cv2.waitKey()

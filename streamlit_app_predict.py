import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64





def load_image():
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        cv2.imwrite("main_image.jpg", opencv_image)
       
    return path, opencv_image
       


	


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return np.array(image)
	

	
def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    #img = cv2.imread(saved_image)
    img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2-30, y-h//2)
        

    img = cv2.rectangle(img, start_pnt, end_pnt, (0,255,0), 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)	
    #st.image(img, caption='Resulting Image')	

    return img



def predict(model, url):
    return model.predict(url, confidence=40, overlap=30).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('TTEPL')

    rf = Roboflow(api_key="FFCwPN6Fvmme9mTdGDLj")
    project_roi = rf.workspace().project("ttepl")
    model_roi = project_roi.version(1).model

    project_digit = rf.workspace().project("ttepl-digit")
    model_digit = project_digit.version(1).model

    image, svd_img = load_image()

    result = st.button('Detect')
    if result:
        results = predict(model_roi, svd_img)
        #results = predict(model2, url)
        print("Prediction Results are...")	
        print(results)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No Object Detected")
        else:
            new_img_pth = results['predictions'][0]['image_path']
            x = int(results['predictions'][0]['x'])
            y = int(results['predictions'][0]['y'])
            w = int(results['predictions'][0]['width'])
            h = int(results['predictions'][0]['height'])
            cl = results['predictions'][0]['class']
            cnf = results['predictions'][0]['confidence']

            img = cv2.imread("main_image.jpg")
            patch = img[y-h//2:y+h//2, x-w//2:x+w//2, :]

            patch = cv2.rotate(patch, cv2.ROTATE_180)

            st.image(patch, caption="ROI")

            det_results = predict(model_digit, patch)

            #st.write(det_results)


            st.write('DETECTION RESULTS')
            
            for i in range (len(det_results['predictions'])):
                
                x_d = int(det_results['predictions'][i]['x'])
                y_d = int(det_results['predictions'][i]['y'])
                w_d = int(det_results['predictions'][i]['width'])
                h_d = int(det_results['predictions'][i]['height'])
                cl_d = det_results['predictions'][i]['class']
                cnf_d = det_results['predictions'][i]['confidence']

            
                patch = drawBoundingBox(patch, x_d, y_d, w_d, h_d, cl_d, cnf_d)

            st.image(patch, caption="Resulting Image")    
           

if __name__ == '__main__':
    main()

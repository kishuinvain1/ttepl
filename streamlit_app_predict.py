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
    txt_start_pnt = (x-w//2, y-h//2-15)
    if cl == 'front-alloy-outer':
        cl = 'front_wheel-front_disc-6_innerclip_bolts'
    elif cl == 'rear-spokes-inner':
        cl = 'rear_wheel-rear_disc-5_innerclip_bolts' 
    elif cl == 'front-alloy-inner':
        cl = 'front_wheel-front_disc-5_innerclip_bolts'       
    
    img = cv2.rectangle(img, start_pnt, end_pnt, (0,255,0), 10)
    img = cv2.putText(img, "KTM200-"+cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2, cv2.LINE_AA)	
    st.image(img, caption='Resulting Image')	
    


def predict(model, url):
    return model.predict(url, confidence=40, overlap=30).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('Tyre Classification')
    rf = Roboflow(api_key="iYDj3AF1byBTN6qkTYwP")
    project = rf.workspace().project("tyre-classification")
    model = project.version(3).model
     
    image, svd_img = load_image()

    result = st.button('Detect')
    if result:
        results = predict(model, svd_img)
        #results = predict(model2, url)
        print("Prediction Results are...")	
        print(results)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No Tyre Detected")
        else:
            new_img_pth = results['predictions'][0]['image_path']
            x = results['predictions'][0]['x']
            y = results['predictions'][0]['y']
            w = results['predictions'][0]['width']
            h = results['predictions'][0]['height']
            cl = results['predictions'][0]['class']
            cnf = results['predictions'][0]['confidence']

            st.write('DETECTION RESULTS')
            st.write('* Model: KTM-200')
            if "front" in cl:    
                st.write('* Front Wheel')
            else:
                st.write('* Rear Wheel')  

            if "alloy" in cl:
                st.write('* Front Disc')
            else:
                st.write('* Rear Disc')  

            if "inner" in cl:
                st.write('* 5 Inner Clip Bolts')  
            else:
                st.write('* 6 Inner Clip Bolts') 

            drawBoundingBox(svd_img,x, y, w, h, cl, cnf)
           

if __name__ == '__main__':
    main()

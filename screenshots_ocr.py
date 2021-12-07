import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import pytesseract
from pytesseract import Output
import tkinter
from tkinter import ttk
import uuid
import time
import threading

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


root = None
canvas = None
canvas_frame = None
pw = None
toolbar = None
tv = None
output_data = None
ocr_confidence_th = 60
axes = None
fig = None

# get grayscale image
def get_grayscale(image):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscaled

# noise removal
def remove_noise(image):
    blurred = cv2.medianBlur(image,5)
    return blurred

#thresholding
def thresholding(image):
    thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(image, kernel, iterations = 1)
    return dilated
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(image, kernel, iterations = 1)
    return eroded

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

#canny edge detection
def canny(image):
    cannied = cv2.Canny(image, 100, 200)
    return cannied

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def preprocess(data, root, filename):
    filepath = os.path.join(root, filename)
    print(f"Preprocessing {filepath}")
    image = cv2.imread(filepath)
    data.append({'image': image, 'title':filename})
    # im_gray = get_grayscale(image)
    # data.append({'image': im_gray, 'title':'Grayscaled'})
    # im_thresh = thresholding(im_gray)
    # data.append({'image': im_thresh, 'title':data[-1]['title']+' Thresholded'})
    # im_opening = opening(im_thresh)
    # data.append({'image': im_opening, 'title':data[-1]['title']+' Opened'})    
    # im_canny = canny(im_opening)
    # data.append({'image': im_canny, 'title':data[-1]['title']+' Canny'})    

    return data

def ocr(data):
    global ocr_confidence_th
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    img = data[0]['image'].copy()
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    d = clean_ocr_data(d, ocr_confidence_th)
    # print(d.keys())
    # print(d['text'])
    # img_txt = pytesseract.image_to_string(preprocessed, config=custom_config)
    img = draw_ocr_boxes(img, d)
    data.append({'image': img, 'title':data[-1]['title']+'\nOCR', 'OCR':d})  
    return data

def draw_ocr_boxes(img, ocr_data, highlight_idx=-1):
    for key in ocr_data.keys():
        (x, y, w, h) = (ocr_data[key]['left'], ocr_data[key]['top'], ocr_data[key]['width'], ocr_data[key]['height'])
        color = (255, 0, 0) if ocr_data[key]['index'] == highlight_idx else (0, 0, 255)
        thickness = 4 if ocr_data[key]['index'] == highlight_idx else 2
        offset = 2 if ocr_data[key]['index'] == highlight_idx else 0
        img = cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), color, thickness)
    return img

def clean_ocr_data(d, ocr_threshold):
    new_dict = {}
    n_boxes = len(d['level'])    
    #Filtering text below threshold
    for i in range(n_boxes):
        if int(d['conf'][i]) > ocr_threshold:
            txt_data = {}
            for key in d.keys():
                txt_data[key] = d[key][i]
            txt_data['index'] = len(new_dict.keys())
            new_dict[d['text'][i]] = txt_data
    return new_dict

def update_preview(output_data, rows=1):
    global fig
    global axes
    global root
    global canvas_frame
    global canvas
    global pw
    root.wm_title(output_data[0]['title'])
    idx = 0
    if canvas is None:
        fig, axes = plt.subplots(nrows=rows, ncols=len(output_data))
        fig.canvas.set_window_title(output_data[0]['title'])
        axes = axes.flatten()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)  # A tk.DrawingArea.
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        canvas.mpl_connect("key_press_event", on_key_press)

    # plt.ion()
    # plt.show()
    for img_data in output_data:
        axes[idx].clear()        
        plt.setp(axes, xticks=[], yticks=[])
        imobj = axes[idx].imshow(img_data['image'])
        img_data['imobj'] = imobj
        axes[idx].set_title(img_data['title'], fontdict={'fontsize':9})
        idx += 1
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    canvas.draw()
    # plt.show()
    # plt.waitforbuttonpress()
    d = output_data[-1]["OCR"]    
    json_tree(tv, '', d)
    root.update()
    tkinter.mainloop()

def setup_window():
    global root
    global canvas
    global canvas_frame
    global toolbar
    global tv
    global output_data
    global axes
    global fig
    global pw
    root = tkinter.Tk()
    pw = tkinter.PanedWindow(root, orient = tkinter.HORIZONTAL)
    pw.pack(fill=tkinter.BOTH, expand=1)

    canvas_frame = tkinter.Frame(root)
    button = tkinter.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tkinter.BOTTOM)
    button = tkinter.Button(master=root, text="Next", command=_next)
    button.pack(side=tkinter.BOTTOM)

    pw.add(canvas_frame)

    tv = ttk.Treeview(root)

    pw.add(tv)
    tv.pack(side=tkinter.RIGHT, fill=tkinter.Y, expand=1)
    tv.bind("<Button-1>", onSingleClick)
    canvas_frame.pack(side=tkinter.LEFT, fill=tkinter.X, expand=1)

def onSingleClick(event):
    global tv
    item = tv.identify('item',event.x,event.y)
    try:
        update_OCR_boxes(highlight_idx=int(tv.item(item,"value")[0]))
    except Exception as e:
        pass

def update_OCR_boxes(highlight_idx=-1):
    global output_data
    img_data = output_data[-1]
    imobj = img_data['imobj']
    img = draw_ocr_boxes(output_data[0]['image'].copy(), img_data['OCR'], highlight_idx)
    imobj.set_data(img)
    plt.draw()


def json_tree(tree, parent, dictionary):
    for key in dictionary:
        uid = uuid.uuid4()
        if isinstance(dictionary[key], dict):
            tree.insert(parent, 'end', uid, text=key, value=dictionary[key]["index"])
            json_tree(tree, uid, dictionary[key])
        elif isinstance(dictionary[key], list):
            tree.insert(parent, 'end', uid, text=key + '[]', value=dictionary[key]["index"])
            json_tree(tree,
                      uid,
                      dict([(i, x) for i, x in enumerate(dictionary[key])]))
        else:
            value = str(dictionary[key])
            if value is None:
                value = 'None'
            tree.insert(parent, 'end', uid, text=f"{key}: {value}")


def on_key_press(event):
    global canvas
    global toolbar
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)


def _quit():
    global root
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    exit()

def _next():
    global root
    root.quit()     # stops mainloop
    

def main():
    global output_data
    ocr_data = []
    print("STARTING!")
    rootdir = "screenshots"
    setup_window()
    for root, dirs, files in os.walk(rootdir):
        ocr_dict = {'root': root}
        # print('--\nroot = ' + root)
        dirs.sort()
        # for subdir in subdirs:
        #     print('\t- subdirectory ' + subdir)
        files = [ file for file in files if file.lower().endswith( ('.jpg','.png', '.jpeg', '.bmp'))]
        for filename in files:
            file_path = os.path.join(root, filename)
            ocr_dict["filename"] = filename
            ocr_dict["filepath"] = file_path
            print('\t- file %s (full path: %s)' % (filename, file_path))
            output_data = []
            output_data = preprocess(output_data, root, filename)
            output_data = ocr(output_data)
            update_preview(output_data)

            # with open(file_path, 'rb') as f:
            #     f_content = f.read()
            #     list_file.write(('The file %s contains:\n' % filename).encode('utf-8'))
            #     list_file.write(f_content)
            #     list_file.write(b'\n')
            


if __name__ == "__main__":
    print("Calling main!")
    main()
# Screenshot OCR Tool

![Screenshot OCR tool example](https://github.com/Gabryxx7/screenshot-ocr/blob/main/example.png?raw=true)

Extracting data from screen time screenshots in iOS and Android.
We are exploring 3 options:
1. Simple OCR with no text position using `pytesseract` and `OpenCV`. We can then try and extract info with `regex`
2. Extract text and its position from each screenshot, classify data according to its position in the screenshot
3. Use YOLOv4 to extract some features from the screenshot and then use those features to train a ML model.
    - https://arxiv.org/abs/2004.10934
    - https://github.com/AlexeyAB/darknet

# Instructions
So far there is not much to do really:
1. Add your screenshots in each folder
2. Run the script and wait for the `tkinter` window to show up
3. The panel on the right lets you explore the text extracted by `pytesseract`
4. Clicking on each top-level text in the tree view will highlight the text on the screenshot in red

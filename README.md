Neural Style transfer

This module takes a content image and style image and then produces a image which will have content from content image but the style will be like style image. Neural Style transfer is based on layers of pretrained VGG19 model.

The directory image contains the content and the style image and the output is produced in same directory whose name will be output.jpg. 

Inorder to perform style transfer on custom inputs, place ur content and style images in images directory and execute main.py in your terminal or command prompt. Type ' python main.py "PATH OF CONTENT IMAGE" "PATH OF STYLE IMAGE" '. Replace the text inside "" with the respective paths of the images.

A test case is provided in the images directory. Here "turtlr.jpg" is the content image and "wave.jpg" is the style image. In the command prompt or terminal type " python main.py ./images/turtle.py ./images/wave.py " and wait for the entire process to get completed. This program runs faster on GPU.

REQUIREMENTS:
1. Tensorflow(GPU Support)
2. PIL
3. Matplotlib
4. Numpy

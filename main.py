import os
import sys
import matplotlib.pyplot as plt
from imgmanipulation import load_image,display
from styletransfer import *
from PIL import Image

content_path = sys.argv[1]
style_path = sys.argv[2]

print("\n \n")
print("=============================================")
print("Style Transfer process started")
best, best_loss = style_transfer(content_path, style_path, iterations=1000)
print("Style Transfer process ended")
print("Output is saved in image directory")
print("=============================================")
print("\n \n")

content = load_image(content_path).astype('uint8')
style = load_image(style_path).astype('uint8')
plt.subplot(1, 2, 1)
display(content, 'Content Image')
plt.subplot(1, 2, 2)
display(style, 'Style Image')
plt.show()

Image.fromarray(best)
best = best.save("./images/output.jpg")
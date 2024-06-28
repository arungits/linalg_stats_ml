import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pic = Image.open("/Users/arun/Downloads/Einstein_tongue.jpg")
pic = np.array(pic)
print(pic.shape)


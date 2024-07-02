import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from PIL import Image

pic = Image.open('Einstein.jpg')

# Convert the pic to numpy array
pic = np.array(pic)

# Show the original image
plt.imshow(pic)
plt.show()

U,S,Vh = LA.svd(pic)

# Use only the top 10% components to reconstruct a low rank version of the image

components = np.arange(0, int(len(S) / 10))

# Print the number of components used for low rank version
print("Number of components used for low rank version:", len(components))

# Low rank version of the image
pic_low_rank = U[:,components]@np.diag(S[components])@Vh[components,:]
plt.imshow(pic_low_rank)
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(seed=42)

COEFFICIENT = 0.5
# Just the one tiny watermark

# input_image = cv2.imread('images\image.jpg')[600:1600, 600:1600, :]
# watermarkImg = cv2.imread('Logos/nasa.png') #[400:1000, 400:1000]

# watermarkImg = cv2.cvtColor(watermarkImg, cv2.COLOR_BGR2GRAY)

# widthToPad = 1 + (input_image.shape[0] - watermarkImg.shape[0]) // 2 
# heightToPad = 1 + (input_image.shape[1] - watermarkImg.shape[1]) // 2
# watermarkImg = np.pad(watermarkImg, ((widthToPad,), (heightToPad,)))

# ret, watermarkImg = cv2.threshold(watermarkImg, 127, 255, cv2.THRESH_BINARY)
# watermarkImg[watermarkImg != 0] = 1
# plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
# # plt.imshow(watermarkImg, cmap="gray")
# plt.show()

#rewritten to get a full image from small watermark
input_image = cv2.imread('Project/Github_04232023\images\image.jpg')
watermarkImgSmall = cv2.imread('Project/Github_04232023\Logos/nasa.png')
watermarkImgSmall = cv2.cvtColor(watermarkImgSmall, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
# plt.imshow(watermarkImg, cmap="gray")
plt.show()

watermarkImg = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)

for i in range(0, watermarkImg.shape[0], watermarkImgSmall.shape[0]):
    for j in range(0, watermarkImg.shape[1], watermarkImgSmall.shape[1]):
        tempWidth = np.minimum(watermarkImgSmall.shape[0] + i, watermarkImg.shape[0] - 1)
        tempHeight = np.minimum(watermarkImgSmall.shape[1] + j, watermarkImg.shape[1] - 1)
        watermarkImg[i:tempWidth, j:tempHeight] = watermarkImgSmall[:tempWidth - i, :tempHeight - j]

ret, watermarkImg = cv2.threshold(watermarkImg, 127, 255, cv2.THRESH_BINARY)
watermarkImg[watermarkImg != 0] = 1
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.imshow(watermarkImg, cmap="gray")
plt.show()
randomBitValue_R = np.random.randint(0,2,256)
randomBitValue_G = np.random.randint(0,2,256)
randomBitValue_B = np.random.randint(0,2,256)
indices = np.arange(256)
LUT_R = dict(zip(indices, randomBitValue_R))
LUT_G = dict(zip(indices, randomBitValue_G))
LUT_B = dict(zip(indices, randomBitValue_B))
# Take in 3 8bit values and return the 8bit value
def watermarkMapping(red_8bit, green_8bit, blue_8bit):
    return LUT_R[red_8bit] ^ LUT_G[green_8bit] ^ LUT_B[blue_8bit]
diffusedError = np.zeros_like(input_image, dtype=np.int8)
output_image = np.zeros_like(input_image, dtype=np.uint8)

for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
        # Do only 1 color channel to mitagate the color change
        randomColor = rng.integers(0,3)
        initalPixelValue = pixelValue = (input_image[i][j] + diffusedError[i][j]) % 256

        # While the mapping does not match the watermark value, change the color channel until it does
        while(watermarkMapping(pixelValue[0], pixelValue[1], pixelValue[2]) != watermarkImg[i][j]):
            pixelValue[randomColor] = (pixelValue[randomColor] + 1) % 256
        
        # Set the output value and get the error for error diffusion
        output_image[i][j] = pixelValue
        outputError = np.uint8(np.sum(initalPixelValue) - np.sum(output_image[i][j]))

        # Diffuse the error if not at right or bottom edge
        if(i <= input_image.shape[0] - 2):
            diffusedError[i+1][j] += np.int8(COEFFICIENT * outputError)
        if(j <= input_image.shape[1] - 2):
            diffusedError[i][j+1] += np.int8(COEFFICIENT * outputError)

                
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
extractedImg = np.zeros((output_image.shape[0], output_image.shape[1]), dtype=np.uint8)
for i in range(extractedImg.shape[0]):
    for j in range(extractedImg.shape[1]):
        extractedImg[i][j] = watermarkMapping(output_image[i][j][0], output_image[i][j][1], output_image[i][j][2])
        # print(watermarkMapping(output_image[i][j][0], output_image[i][j][1], output_image[i][j][2]))

# extractedImg[extractedImg == 1] = 255
plt.imshow(extractedImg, cmap="gray")
plt.show()

print(f" Difference between watermark and extracted: {np.sum(watermarkImg - extractedImg)}")

## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: 
Import necessary libraries (cv2, NumPy, Matplotlib) for image loading, filtering, and visualization.

### Step 2: 
Load the image using cv2.imread() and convert it to RGB format using cv2.cvtColor() for proper display in Matplotlib.

### Step 3: 
Apply different filters:
1. Averaging Filter: Define an averaging kernel using np.ones() and apply it to the image using cv2.filter2D().
2. Weighted Averaging Filter: Define a weighted kernel (e.g., 3x3 Gaussian-like) and apply it with cv2.filter2D().
3. Gaussian Filter: Use cv2.GaussianBlur() to apply Gaussian blur.
4. Median Filter: Use cv2.medianBlur() to reduce noise.
5. Laplacian Operator: Use cv2.Laplacian() to apply edge detection.
    

### Step 4: 
Display each filtered image using plt.subplot() and plt.imshow() for side-by-side comparison of the original and processed images.

### Step 5: 
Save or show the images using plt.show() after applying each filter to visualize the effects of smoothing and sharpening.

## Program:

```
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1 = cv2.imread("dog.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
```

### 1. Smoothing Filters

#### i) Using Averaging Filter
```
kernel = np.ones((11, 11), np.float32) / 121
averaging_image = cv2.filter2D(image2, -1, kernel)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(averaging_image)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()

```
#### ii) Using Weighted Averaging Filter
```
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

weighted_average_image = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(weighted_average_image)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

```
#### iii) Using Gaussian Filter
```
gaussian_blur = cv2.GaussianBlur(image2, (11, 11), 0)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()

```

#### iv) Using Median Filter
```
median_blur = cv2.medianBlur(image2, 11)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
#### i) Using Laplacian Kernal
```
sharpened_image = cv2.filter2D(smoothed_image, -1, kernel2)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(sharpened_image)
plt.title("Sharpened Image (Laplacian Kernel)")
plt.axis("off")
plt.show()
```
#### ii) Using Laplacian Operator
```
laplacian = cv2.Laplacian(image2, cv2.CV_64F)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()
```
## OUTPUT:
### 1. Smoothing Filters

![Screenshot 2024-10-01 112413](https://github.com/user-attachments/assets/6efb4b7f-ec64-4c42-bd7e-2771ebb9b9af)

i) Using Averaging Filter

![Screenshot 2024-10-01 112448](https://github.com/user-attachments/assets/37782621-1fb2-46c0-9f9e-a802a15d3c62)

ii)Using Weighted Averaging Filter

![Screenshot 2024-10-01 112519](https://github.com/user-attachments/assets/867edbd4-59b1-46a7-acf5-894857166144)

iii)Using Gaussian Filter

![Screenshot 2024-10-01 112541](https://github.com/user-attachments/assets/554a2785-1e0c-4986-8385-f89be0b2c4a6)

iv) Using Median Filter

![Screenshot 2024-10-01 112601](https://github.com/user-attachments/assets/7b5a69aa-5589-4685-a4ec-611cb6c04a09)

### 2. Sharpening Filters

i) Using Laplacian Kernal

![Screenshot 2024-10-01 112622](https://github.com/user-attachments/assets/61c52024-9aae-45fe-80a6-dbe8090675ba)

ii) Using Laplacian Operator

![Screenshot 2024-10-01 112643](https://github.com/user-attachments/assets/696e41a8-4d0b-47d9-beac-b0a3c7e8227e)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.

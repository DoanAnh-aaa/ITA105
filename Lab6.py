import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def load_random_images(folder, n):
    files = os.listdir(folder)
    chosen = random.sample(files, min(n, len(files)))
    images = []
    for f in chosen:
        img = cv2.imread(os.path.join(folder, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def apply_rotation(img, angle_range):
    angle = random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_brightness(img, percent_range):
    percent = random.uniform(-percent_range, percent_range)
    factor = 1.0 + percent
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def bai1(folder):
    orig_images = load_random_images(folder, 5)
    aug_images = []

    for img in orig_images:
        img = cv2.resize(img, (224, 224))

        if random.choice([True, False]):
            img = cv2.flip(img, 1)
        img = apply_rotation(img, 15)
        img = apply_brightness(img, 0.20)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32) / 255.0
        aug_images.append(img)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(orig_images[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(aug_images[i], cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def bai2(folder):
    img = load_random_images(folder, 1)[0]
    img = cv2.resize(img, (224, 224))
    img = apply_rotation(img, 10)
    img = apply_brightness(img, 0.15)
    noise = np.random.normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_norm = img_gray.astype(np.float32) / 255.0
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Augmented")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img_norm, cmap='gray')
    plt.title("Normalized")
    plt.axis('off')
    plt.show()

def bai3(folder):
    orig = load_random_images(folder, 1)[0]
    grid = []
    for _ in range(9):
        img = cv2.resize(orig, (224, 224))
        if random.choice([True, False]):
            crop = random.randint(150, 200)
            x = random.randint(0, 224-crop)
            y = random.randint(0, 224-crop)
            img = img[y:y+crop, x:x+crop]
            img = cv2.resize(img, (224, 224))
        if random.choice([True, False]):
            img = cv2.flip(img, 1)
        img = apply_rotation(img, 15)
        img = img.astype(np.float32) / 255.0
        grid.append(img)
    fig, axes = plt.subplots(3,3, figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grid[i])
        ax.axis('off')
    plt.show()

def bai4(folder):
    img = load_random_images(folder, 1)[0]
    img = cv2.resize(img, (224, 224))
    aug_list = []
    for _ in range(3):
        aug = img.copy()
        aug = apply_rotation(aug, 15)
        if random.choice([True, False]):
            aug = cv2.flip(aug, 1)
        aug = apply_brightness(aug, 0.20)
        aug = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY)
        aug = aug.astype(np.float32) / 255.0
        aug_list.append(aug)
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    for i in range(3):
        plt.subplot(1,4,i+2)
        plt.imshow(aug_list[i], cmap='gray')
        plt.title(f"Aug {i+1}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    bai1('data/apartment')
    bai2('data/vehicles')
    bai3('data/fruits')
    bai4('data/interior')

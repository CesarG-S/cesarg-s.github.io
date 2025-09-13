---
layout: default
permalink: /project1/
title: CS180 - Project 1
---
## Introduction

The goal of this project is to align three images, one representing the red color channel, one the blue, and one the green, and create a colored image. The three images are not perfectly aligned, and borders on the ends of the photos prevent me from simply stacking the images and calling it a day. There exists a displacement for each color channel that lines up each color frame in a way where the original color picture is clear and free of major imperfections. The goal is to score each displacement and pick the ones with the best score.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/naive_stack.jpg" alt="Naive image stacking" class="img-33">
        <figcaption>Naive stacking</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_aligned_euclid.jpg" alt="Stacking with Euclidean Distance" class="img-33">
        <figcaption>Stacking with Euclidean Distance</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_aligned_euclid.jpg" alt="Stacking with Euclidean Distance" class="img-33">
        <figcaption>Stacking with Euclidean Distance</figcaption>
    </figure>
</div>

## Preprocessing

Before we get into the actual displacement calculations, it would be nice if we could do something about those borders. The major blemishes are concentrated around the edge of the image, so I will simply crop the image to remove said blemishes. This shouldn't affect my calculations - if anything, it would improve them - since all I am trying to calculate is an (x,y) displacement value that I can apply back to the original image. I opted to crop out the first and last 15% of the horizontal and vertical image, or in other words, I preserved the inner 70% of the image.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/naive_stack_crop05.jpg" alt="Naive image stacking" class="img-20">
        <figcaption>Crop 5%</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/naive_stack_crop10.jpg" alt="Naive image stacking" class="img-20">
        <figcaption>Crop 10%</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/naive_stack_crop15.jpg" alt="Naive image stacking" class="img-20">
        <figcaption>Crop 15%</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/naive_stack_crop20.jpg" alt="Naive image stacking" class="img-20">
        <figcaption>Crop 20%</figcaption>
    </figure>
</div>

After this, once I calculate my displacements, I apply them to the *original* image.

## Single-scale Euclidean Distance
The process for calculating the proper displacements and creating a colored image using the Euclidean Distance is as follows:

1. Import the image as a float
2. Crop the outside 15% of the blue, green, and red channels. 
3. Calculate the displacements needed to align the cropped red and blue channels using the cropped green channel as the reference.
4. Align the original red and blue channels with the original green channel using the displacements from the cropped channels.
5. Combine the color channels back into a matrix and display the image.
6. Optional. Crop the colored image to remove the ugly miscolored borders.
7. Relish in the beauty of your wonderful colored image.  

<br>
For step 3, the calculation step involves taking either the red or blue channel, applying an offset in both the x and y directions (with np.roll), then taking the difference between the offset channel and the green reference channel element-wise for every pixel value, squaring that difference, and adding up all the squared differences. I keep track of the minimum sum of squared distances and the offset used to achieve that sum for all offsets. My offsets are [-16,16], meaning I take the entire red channel, for example, shift all the pixels X amount to the left/right and Y amount up/down, and then sum the squared distances, for all values of X and Y between -16 and 16.

My distance calculation for step 3: <code>distance = np.sqrt(np.sum(np.sum((reference - currentTester) ** 2, axis=1)))</code> <br>
        - reference is the green color channel and currentTester is the current red or blue color channel offset in X and Y -

For step 4, I simply take the displacements from step 3 and use np.roll to shift the original red and blue channels.

For steps 5 and 6, I used the same array slicing technique to delete the outer pixels of the image. I find removing 5% of the outer pixels is good enough. After that, I stack the color channels into a single array and display that as my final colored image.

I ran into a few issues with rolling in the wrong axis and cropping the wrong image matrices. It is important to use the axis=(1,0) parameter in np.roll if your displacements are in the format (x,y), where positive x means shifting right and positive y means shifting down.

Using this technique, I was able to generate images for cathedral.jpg, monastery.jpg, and tobolsk.jpg

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_aligned_euclid.jpg" alt="Cathedral with image borders using Euclidean Distance" class="img-20">
        <figcaption>cathedral.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/monastery_aligned_euclid.jpg" alt="Monastery with image borders using Euclidean Distance" class="img-20">
        <figcaption>monastery.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/tobolsk_aligned_euclid.jpg" alt="Tobolsk with image borders using Euclidean Distance" class="img-20">
        <figcaption>tobolsk.jpg</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_aligned_euclid_cropped.jpg" alt="Cathedral without image borders using Euclidean Distance" class="img-33">
        <figcaption>cathedral.jpg cropped</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/monastery_aligned_euclid_cropped.jpg" alt="Monastery without image borders using Euclidean Distance" class="img-33">
        <figcaption>monastery.jpg cropped</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/tobolsk_aligned_euclid_cropped.jpg" alt="Tobolsk without image borders using Euclidean Distance" class="img-33">
        <figcaption>tobolsk.jpg cropped</figcaption>
    </figure>
</div>

In my opinion, the procedure worked very well for cathedral.jpg, did well on monastery.jpg, but may not have done perfectly on tobolsk.jpg. In particular, the alignment of red and blue channels appears to be slightly off and is most apparent in the wooden planks floating on the water. There also appears to be a reddish vignette in monastery.jpg, but that may be a consequence of how the photos were taken, rather than my algorithm failing to align the images.

## Single-scale Normalized Cross-Correlation
The process for NCC was the same as for using the Euclidean Distance except for step 3.

The only difference was the calculation of the distance, or in this case, similarity of vectors for the two color channels. The problem also becomes a maximization problem, rather than a minimization. That is, we wish to find the displacement that results in the greatest normalized dot product across the displaced color channels and the reference.

My distance calculation for step 3: <code>distance = np.sum(reference*currentTester) / (np.sqrt(np.sum(reference**2)) * np.sqrt(np.sum(currentTester**2)))</code>

I generated the same images as before using NCC:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_aligned_ncc.jpg" alt="Cathedral with image borders using NCC" class="img-20">
        <figcaption>cathedral.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/monastery_aligned_ncc.jpg" alt="Monastery with image borders using NCC" class="img-20">
        <figcaption>monastery.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/tobolsk_aligned_ncc.jpg" alt="Tobolsk with image borders using NCC" class="img-20">
        <figcaption>tobolsk.jpg</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_aligned_ncc_cropped.jpg" alt="Cathedral without image borders using NCC" class="img-33">
        <figcaption>cathedral.jpg cropped</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/monastery_aligned_ncc_cropped.jpg" alt="Monastery without image borders using NCC" class="img-33">
        <figcaption>monastery.jpg cropped</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/tobolsk_aligned_ncc_cropped.jpg" alt="Tobolsk without image borders using NCC" class="img-33">
        <figcaption>tobolsk.jpg cropped</figcaption>
    </figure>
</div>

Perhaps unsurprisingly, these are the same images, just created with a different algorithm.

Out of curiosity, I ran the NCC algorithm on church.tif, to see how long it would take. After 3 minutes and 26 seconds, I generated this image:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/naive_church.jpg" alt="Church with image borders using NCC" class="img-50">
        <figcaption>church.tif saved as .jpg</figcaption>
    </figure>
</div>

It seems like the image is in need of some fine-tuning. It is possible that the right displacement wasn't within the range I set for the alignment-finding function. The next step is to use image pyramids to speed up this process by running iterations of NCC on smaller-sized versions of the large image to more easily find the best alignment in the large image.

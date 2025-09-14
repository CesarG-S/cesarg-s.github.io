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
        <img src="/assets/images/proj1/cathedral_aligned_ncc.jpg" alt="Stacking with NCC" class="img-33">
        <figcaption>Stacking with Normalized Cross-Correlation</figcaption>
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

## NCC with Image Pyramids
The algorithm changes very slightly when introducing image pyramids. Before step 3 of the single-scale algorithm, I create a set of image pyramids of the reference channel and one of the other color channels. To do that, simply take your original color channel, shrink it by a factor of 2 with anti-aliasing, then add it to an array. Take that scaled-down image and shrink it again, to create the next entry of the pyramid. Repeat this process until you reach an appropriate size threshold. 

Once I have the image pyramids for the channels I'm working with, I have to reverse the order, because I inserted images from largest to smallest. I take the two smallest images of the reference (green) channel pyramid, and either the red or blue channel pyramid and crop them to eliminate any potential artifacts caused by np.roll. This is because I generate displacements in x and y over a displacement window to shift the test channel (red or blue) and I use np.roll to actually shift the elements. This process would also work with single-scale NCC, but I didn't think about it at the time. np.roll loops back around, so if I crop away the looped-around part, I will be comparing only the parts that matter. Since I'm cropping the test channel, I also have to crop the reference channel so I can perform NCC. I keep track of the highest NCC score and the corresponding displacement once I've gone through the entire displacement window.

Once I've gotten the best displacement for that image, I move on to the next image in each pyramid, but I have to scale the displacement in the next iteration because the images are twice the size now. The intuition is something like: pixel number 4 of the original image is now pixel 8, so displacement 2 becomes displacement 4. After the displacement is doubled, we run the loop again, but over a smaller displacement window (half the size). The idea is the previous displacement is a pretty good guess as to where the actual best displacement is, but since the pixels now create a higher resolution image, you have to consider the "in-betweens" of what would have been the smaller image because the true best displacement could be hiding in the gap previously blurred by the smaller image, if that makes sense.

You repeat the algorithm until you reach the largest image. Now that you have roughly the right displacement, you test the final surrounding area with a small displacement window, and hopefully, the algorithm returns the best displacements in x and y for the original image.

## The Full Algorithm
This is every single step I take to get to the final image. First, I read in my image. Then I convert the image to float values with img_as_float. The next step is to crop the tall 3-panel image to remove the borders. I actually went with 30% off all sides instead of 15% because it performed a lot faster and seemed to achieve very good results.

The next step is to calculate the displacements. As described before, I create image pyramids for the green channel as a reference and the red or blue channel as the test. The alignment function takes the pyramids, from smallest image to largest image, rolls the test channel over a window of displacements in x and y, and crops the centers before calculating the NCC. The displacement with the max NCC is tracked for each image, and used in the next iteration, scaled up, alongside a halved displacement search window, to efficiently search the next-largest image while doing less work. The process repeats until the largest image is scanned through. The alignment function returns the best displacement for each channel.

Once the displacements are obtained, I simply roll each channel and its corresponding displacement using np.roll. Once I have the correctly-aligned r, g, and b channels, I clip them to have float values between 0 and 1. I had to do this because there was some floating point voodoo going on, and my 1 was 1.0000004 or something. Once the values are clipped, I just stack the r, g, and b channels on top of each other with np.dstack, but not before cropping for the last time the ugly colors on the ends caused by stacking the offset color channels.

At this point, I can save the image as a file. For this website, I saved the uncropped and cropped versions of each image.

## A Confession
I'm writing this after I've already applied an old version of my algorithm to most images. it wasn't until I arrived at self_portrait.tif that I realized my algorithm was wrong. You see, I wasn't considering the fact that the best displacement needs to be rescaled as you move on with the next element of the image pyramid. I worked around this by implementing a minimum displacement window value, which was high enough to essentially perform a naive NCC at the last image of the pyramid. The displacement got close in the right approximate direction, but ultimately this method suffered from being super slow as a result of almost throwing away the intermediate steps and just doing all the work at the last step. It took me reading through a lot of Ed posts to realize this crucial implementation detail. It's crazy to look back, because my code was taking 15 seconds to generate an image. Then I bumped up the parameters to make it more accurate, and that took 45 seconds to generate an image. However, now I generate an image in about 2 seconds, so I feel both gutted and relieved. On one hand, I wasted a lot of time, but on the other hand, the self_portrait.tif (and everyone else) is wonderfully aligned.

## Results
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/church_pyramid.jpg" alt="Church with image borders using pyramid NCC" class="img-50">
        <figcaption>church.tif</figcaption>
        <figcaption>b_displace = (-4,-25); r_displace = (-8,33)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/church_pyramid_cropped.jpg" alt="Church without image borders using pyramid NCC" class="img-50">
        <figcaption>church.tif (cropped)</figcaption>
        <figcaption>b_displace = (-4,-25); r_displace = (-8,33)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/emir_pyramid.jpg" alt="Emir with image borders using pyramid NCC" class="img-50">
        <figcaption>emir.tif</figcaption>
        <figcaption>b_displace = (-24,-48); r_displace = (17,58)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/emir_pyramid_cropped.jpg" alt="Emir without image borders using pyramid NCC" class="img-50">
        <figcaption>emir.tif (cropped)</figcaption>
        <figcaption>b_displace = (-24,-48); r_displace = (17,58)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/harvesters_pyramid.jpg" alt="Harvesters with image borders using pyramid NCC" class="img-50">
        <figcaption>harvesters.tif</figcaption>
        <figcaption>b_displace = (-17,-59); r_displace = (-3,64)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/harvesters_pyramid_cropped.jpg" alt="Harvesters without image borders using pyramid NCC" class="img-50">
        <figcaption>harvesters.tif (cropped)</figcaption>
        <figcaption>b_displace = (-17,-59); r_displace = (-3,64)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/icon_pyramid.jpg" alt="Icon with image borders using pyramid NCC" class="img-50">
        <figcaption>icon.tif</figcaption>
        <figcaption>b_displace = (-18,-41); r_displace = (5,49)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/icon_pyramid_cropped.jpg" alt="Icon without image borders using pyramid NCC" class="img-50">
        <figcaption>icon.tif (cropped)</figcaption>
        <figcaption>b_displace = (-18,-41); r_displace = (5,49)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/italil_pyramid.jpg" alt="Italil with image borders using pyramid NCC" class="img-50">
        <figcaption>italil.tif</figcaption>
        <figcaption>b_displace = (-21,-38); r_displace = (15,39)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/italil_pyramid_cropped.jpg" alt="Italil without image borders using pyramid NCC" class="img-50">
        <figcaption>italil.tif (cropped)</figcaption>
        <figcaption>b_displace = (-21,-38); r_displace = (15,39)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/lastochikino_pyramid.jpg" alt="Lastochikino with image borders using pyramid NCC" class="img-50">
        <figcaption>lastochikino.tif</figcaption>
        <figcaption>b_displace = (2,3); r_displace = (-6,78)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/lastochikino_pyramid_cropped.jpg" alt="Lastochikino without image borders using pyramid NCC" class="img-50">
        <figcaption>lastochikino.tif (cropped)</figcaption>
        <figcaption>b_displace = (2,3); r_displace = (-6,78)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/lugano_pyramid.jpg" alt="Lugano with image borders using pyramid NCC" class="img-50">
        <figcaption>lugano.tif</figcaption>
        <figcaption>b_displace = (15,-39); r_displace = (-13,53)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/lugano_pyramid_cropped.jpg" alt="Lugano without image borders using pyramid NCC" class="img-50">
        <figcaption>lugano.tif (cropped)</figcaption>
        <figcaption>b_displace = (15,-39); r_displace = (-13,53)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/melons_pyramid.jpg" alt="Melons with image borders using pyramid NCC" class="img-50">
        <figcaption>melons.tif</figcaption>
        <figcaption>b_displace = (-10,-84); r_displace = (3,97)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/melons_pyramid_cropped.jpg" alt="Melons without image borders using pyramid NCC" class="img-50">
        <figcaption>melons.tif (cropped)</figcaption>
        <figcaption>b_displace = (-10,-84); r_displace = (3,97)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/self_portrait_pyramid.jpg" alt="Self portrait with image borders using pyramid NCC" class="img-50">
        <figcaption>self_portrait.tif</figcaption>
        <figcaption>b_displace = (-29,-78); r_displace = (8,98)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/self_portrait_pyramid_cropped.jpg" alt="Self portrait without image borders using pyramid NCC" class="img-50">
        <figcaption>self_portrait.tif (cropped)</figcaption>
        <figcaption>b_displace = (-29,-78); r_displace = (8,98)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/siren_pyramid.jpg" alt="Siren with image borders using pyramid NCC" class="img-50">
        <figcaption>siren.tif</figcaption>
        <figcaption>b_displace = (5,-49); r_displace = (-19,46)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/siren_pyramid_cropped.jpg" alt="Siren without image borders using pyramid NCC" class="img-50">
        <figcaption>siren.tif (cropped)</figcaption>
        <figcaption>b_displace = (5,-49); r_displace = (-19,46)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/three_generations_pyramid.jpg" alt="Three generations with image borders using pyramid NCC" class="img-50">
        <figcaption>three_generations.tif</figcaption>
        <figcaption>b_displace = (-18,-49); r_displace = (-3,61)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/three_generations_pyramid_cropped.jpg" alt="Three generations without image borders using pyramid NCC" class="img-50">
        <figcaption>three_generations.tif (cropped)</figcaption>
        <figcaption>b_displace = (-18,-49); r_displace = (-3,61)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_pyramid.jpg" alt="Cathedral with image borders using pyramid NCC" class="img-50">
        <figcaption>cathedral.jpg</figcaption>
        <figcaption>b_displace = (-2,-5); r_displace = (1,7)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/cathedral_pyramid_cropped.jpg" alt="Cathedral without image borders using pyramid NCC" class="img-50">
        <figcaption>cathedral.jpg (cropped)</figcaption>
        <figcaption>b_displace = (-2,-5); r_displace = (1,7)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/monastery_pyramid.jpg" alt="Monastery with image borders using pyramid NCC" class="img-50">
        <figcaption>monastery.jpg</figcaption>
        <figcaption>b_displace = (-2,3); r_displace = (1,6)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/monastery_pyramid_cropped.jpg" alt="Monastery without image borders using pyramid NCC" class="img-50">
        <figcaption>monastery.jpg (cropped)</figcaption>
        <figcaption>b_displace = (-2,3); r_displace = (1,6)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/tobolsk_pyramid.jpg" alt="Tobolsk with image borders using pyramid NCC" class="img-50">
        <figcaption>tobolsk.jpg</figcaption>
        <figcaption>b_displace = (-3,-3); r_displace = (1,4)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/tobolsk_pyramid_cropped.jpg" alt="Tobolsk without image borders using pyramid NCC" class="img-50">
        <figcaption>tobolsk.jpg (cropped)</figcaption>
        <figcaption>b_displace = (-3,-3); r_displace = (1,4)</figcaption>
    </figure>
</div>

## Bonus images
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/embroidery_pyramid.jpg" alt="Embroidery with image borders using pyramid NCC" class="img-50">
        <figcaption>embroidery.tif</figcaption>
        <figcaption>b_displace = (-8,-73); r_displace = (8,85)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/embroidery_pyramid_cropped.jpg" alt="Embroidery without image borders using pyramid NCC" class="img-50">
        <figcaption>embroidery.tif (cropped)</figcaption>
        <figcaption>b_displace = (-8,-73); r_displace = (8,85)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/napoleon_pyramid.jpg" alt="Napoleon with image borders using pyramid NCC" class="img-50">
        <figcaption>napoleon.tif</figcaption>
        <figcaption>b_displace = (-5,-63); r_displace = (-8,70)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/napoleon_pyramid_cropped.jpg" alt="Napoleon without image borders using pyramid NCC" class="img-50">
        <figcaption>napoleon.tif (cropped)</figcaption>
        <figcaption>b_displace = (-5,-63); r_displace = (-8,70)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/roses_pyramid.jpg" alt="Roses with image borders using pyramid NCC" class="img-50">
        <figcaption>roses.tif</figcaption>
        <figcaption>b_displace = (-20,-30); r_displace = (16,52)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj1/roses_pyramid_cropped.jpg" alt="Roses without image borders using pyramid NCC" class="img-50">
        <figcaption>roses.tif (cropped)</figcaption>
        <figcaption>b_displace = (-20,-30); r_displace = (16,52)</figcaption>
    </figure>
</div>
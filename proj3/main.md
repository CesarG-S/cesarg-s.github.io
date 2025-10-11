---
layout: default
permalink: /project3/
title: CS180 - Project 3
---
## Introduction
The goal of this project is to stitch images together in a convincing manner.

## Photo Gallery
The first thing you need when stitching photos together are the photos. These photos were captured by standing still, taking the first photo, then rotating the phone in place while doing my best not to move it from where it stood when the first photo was taken. This ensures that the relationship between the first and second image can be described with a homography.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/livingroom_base.jpg" alt="Living room" class="img-33">
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/livingroom_far.jpg" alt="Living room turned" class="img-33">
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/tree_base.jpg" alt="Trees" class="img-33">
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/tree_far.jpg" alt="Trees turned" class="img-33">
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/ducks_base.jpg" alt="Ducks" class="img-33">
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/ducks_far.jpg" alt="Ducks turned" class="img-33">
    </figure>
</div>

## Image Homography
To go from one image to the other, we must calculate the homography matrix H, such that for a point p in image 1, Hp = p' for p' in image 2. At this stage, we have to manually obtain the points in image 1 that match with points in image 2. I used ginput and zoomed in to precisely mark the points that match.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/livingroom_base_labeled.jpg" alt="Living room labeled" class="img-33">
        <figcaption>im1 labeled</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/livingroom_far_labeled.jpg" alt="Living room turned labeled" class="img-33">
        <figcaption>im2 labeled</figcaption>
    </figure>
</div>

With these points, we can solve for the elements of the homography matrix. The known values are the x's, the y's, the u's, and the v's, and the unknowns are the h's. Points from image 1 are (x,y) pairs and points from image 2 are (u,v) pairs. Points with the same subscript are correspondences.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/homography_syseq.png" alt="System of equations" class="img-33">
    </figure> ➤➤➤➤➤➤➤➤ <br> (via least squares)

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/homography_matrix.png" alt="Homography matrix" class="img-33">
    </figure>
</div>

As you may have noticed, I have more than 4 correspondences. The system of equations follows the same pattern, but with more rows than unknowns. To solve this system of equations, I use least squares. This gives me the h's with which I may build my homography matrix H. As a final step, I append a 1 and reshape into a 3x3 matrix.

A crucial detail to understand in this calculation is in my case, given my function computeH(im1pts, im2pts), image 1 represents the "source" image, or where the pixels come from, and image 2 represents the "destination" image, or where the pixels will "warp to" after the calculations are made. This is an important distinction to make, because if you are trying to warp image 2 to look like image 1, then you need to find the correspondences and compute H using computeH(im2pts, im1pts). This little detail caused me a lot of headache while debugging.

With that in mind, using the above images im1 and im2 as inputs, and after manually selecting correspondences, we calculate the homography matrix:

```python
computeH(im2pts, im1pts) # IMPORTANT: we want to warp im2 to the shape of im1

# ========= Output ========= #
im1 points: # (x,y) values
 [
    [24, 379], [38, 379], [24, 397], [38, 397], 
    [79, 313], [149, 315], [78, 410], [147, 413], 
    [186, 343], [256, 343], [187, 440], [256, 441], 
    [299, 401], [326, 401], [299, 427], [326, 428], 
    [270, 565], [257, 576], [181, 494], [173, 556], 
    [124, 555], [121, 555], [210, 763], [210, 749]
]
im2 points: # (u,v) values
 [
    [410, 403], [421, 402], [410, 418], [421, 418], 
    [454, 339], [515, 332], [453, 427], [512, 426], 
    [550, 354], [622, 346], [550, 450], [621, 447], 
    [670, 402], [702, 399], [669, 430], [702, 429], 
    [638, 578], [623, 589], [544, 503], [537, 564], 
    [492, 560], [489, 560], [578, 775], [578, 761]
]
Homography matrix:
 [[ 2.14537967e+00 -8.68234276e-03 -8.37155459e+02]
 [ 7.29597696e-01  1.86290176e+00 -4.38410497e+02]
 [ 1.46265510e-03  3.31163516e-05  1.00000000e+00]]
```

## Performing Image Warping
Now that we have our homography matrix, we can perform inverse mapping on our canvas and sample the image we will warp (im2). But to determine the size of the canvas, we can use our homography matrix to map the corners to their locations on the warped image. The size of the bounding box is equal to the difference between the maximum y and minimum y (height) and the difference between the maximum x and minimum x (width). We keep track of the minimum x and y values of the corners for later use. 

I iterate over all points in the warped image canvas and create an augmented vector in the form (x, y, 1) (transposed) for use in inverse mapping. By matrix multiplying with H inverse, then scaling such that the last element is equal to 1, you get a set of points (x', y') for which to sample from the source image (image 2 in this case). Do keep in mind that the original vector (x, y, 1) can have negative values, and that's fine, because after multiplying with H inverse, we land back within our source image. We finally insert the sampled points into an offset of the warped image. If the point (x', y') we get lands outside the source image bounds, we can set the value of that pixel in the warped image to have an alpha value of 0.

You might ask, what is the offset I use? The answer is the minimum x and minimum y values calculated earlier from finding the corner homography correspondences. This is because if we defined our bounding box to be the same size as the original image, then the stretch would line up with the destination image, but a lot of the warped image would be missing, because technically it lands outside the bounding box. The minimum x and y values can be negative, and by essentially expanding the bounding box, we move the "origin" to start at the minimum x and y values. That's why when we perform inverse mapping, we ultimately have to insert the sampled pixel value at an offset in the warped image canvas.

Of course, you can also vary the way you sample. I used both nearest neighbor sampling and bilinear sampling, and the results were pretty good regardless.

Before we move on to warping and stitching images together, it is good to double check that the homography code works fine. I perform image rectification on images with rectangular objects of known dimensions. I took these photos at funny angles to make straightening them out more challenging. I recommend opening each image in a new tab to really see the difference.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/poster_18x24.jpg" alt="Poster 18x24" class="img-20">
        <figcaption>18"x24" poster frame</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/rectified_NN_poster_18x24.jpg" alt="Rectified poster NN 18x24" class="img-33">
        <figcaption>Rectified poster (nearest neighbor)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/rectified_bilinear_poster_18x24.jpg" alt="Rectified poster bilinear 18x24" class="img-33">
        <figcaption>Rectified poster (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/poster_11.25x17.25.jpg" alt="Poster 11.25x17.25" class="img-20">
        <figcaption>11.25"x17.25" poster frame</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/rectified_NN_poster_11.25x17.25.jpg" alt="Rectified poster NN 11.25x17.25" class="img-33">
        <figcaption>Rectified poster (nearest neighbor)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/rectified_bilinear_poster_11.25x17.25.jpg" alt="Rectified poster bilinear 11.25x17.25" class="img-33">
        <figcaption>Rectified poster (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/book_6x8.5.jpg" alt="Book 6x8.5" class="img-20">
        <figcaption>6"x8.5" book</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/rectified_NN_book_6x8.5.jpg" alt="Rectified book NN 6x8.5" class="img-33">
        <figcaption>Rectified book (nearest neighbor)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/rectified_bilinear_book_6x8.5.jpg" alt="Rectified book bilinear 6x8.5" class="img-33">
        <figcaption>Rectified book (bilinear)</figcaption>
    </figure>
</div>

Looks good. I just picked the rectangle nearest to 500x500 pixels with the same aspect ratio as the object in the photo, so the output sizes aren't necessarily representative of the object's real size. There is a clear improvement when using bilinear interpolation. The time to compute each warped image is fairly similar; there was a difference of 5-6 seconds between nearest neighbor and bilinear, with about 9 seconds vs 15 seconds for images roughly 2000x2500 pixels in size, respectively. I think this difference is small enough to make bilinear the clear winner.

## Image Stitching and Blending
Now for the real fun. I have already described the algorithm above, but basically, after you create the warped image, you need to put both image 1 and the warped image 2 onto the same canvas. We know that we shifted each output pixel by some offset when forming the warped image. When deciding where to place image 1, if we place the warped image at the origin, then we simply translate image 1 by that offset, and it will be placed at the correct overlap. However, simply placing each image over the other may produce a visible seam.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/livingroom_NN_stitch.png" alt="Living room NN stitch" class="img-33">
        <figcaption>Image stitching with NN</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/livingroom_bilinear_stitch.png" alt="Living room bilinear stitch" class="img-33">
        <figcaption>Image stitching with bilinear</figcaption>
    </figure>
</div>

To handle this, I created an alpha channel for each image before warping in the shape of a Gaussian centered at the middle of each image. After some experimentation, picking sigma values of height/5 and width/5 for the y Gaussian and the x Gaussian, respectively, helped hide the seam rather well. Perform an outer product, and normalize so that the center is 1, and you have a Gaussian alpha channel.

During the image stacking step, I found where image 1 and warped image 2 overlap, took the alpha channels, and performed alpha blending according to:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/alpha_blending.jpg" alt="Alpha blending" class="img-33">
        <figcaption>(from Wikipedia "Alpha Compositing")</figcaption>
    </figure>
</div>

Here, alpha o is the combined alpha channel from image 1 (alpha a) and image 2 (alpha b). Co is the combined image after blending image 1 (Ca) and image 2 (Cb) with the previously computed alpha a, alpha b, and alpha o. I also have an extra measure in place to ensure we are not dividing by 0.

We take this blended image Co and alpha o and place them in that overlap area on the canvas. Now that we have this blended image, we can go ahead and remove all transparency effects from the rest of the image before displaying the final product. That procedure is also simple: set all non-zero alpha values to 1, leaving 0 alpha pixels untouched. The results speak for themselves:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_livingroom_base.jpg" alt="Living room base alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/livingroom_base.jpg" alt="Living room base" class="img-33 default-image">
        </div>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_livingroom_far.jpg" alt="Living room far alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/livingroom_far.jpg" alt="Living room far" class="img-33 default-image">
        </div>
    </figure> ➤

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_NN_livingroom_base_livingroom_far.jpg" alt="Stitched living room NN" class="img-33">
        <figcaption>Stitched living room (NN)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_livingroom_base_livingroom_far.jpg" alt="Stitched living room bilinear" class="img-33">
        <figcaption>Stitched living room (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_ducks_far.jpg" alt="Duck far alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/ducks_far.jpg" alt="Duck far" class="img-33 default-image">
        </div>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_ducks_base.jpg" alt="Duck base alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/ducks_base.jpg" alt="Duck base" class="img-33 default-image">
        </div>
    </figure> ➤

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_NN_ducks_far_ducks_base.jpg" alt="Stitched duck NN" class="img-33">
        <figcaption>Stitched ducks (NN)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_ducks_far_ducks_base.jpg" alt="Stitched duck bilinear" class="img-33">
        <figcaption>Stitched ducks (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_tree_far.jpg" alt="Tree far alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/tree_far.jpg" alt="Tree far" class="img-33 default-image">
        </div>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_tree_base.jpg" alt="Tree base alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/tree_base.jpg" alt="Tree base" class="img-33 default-image">
        </div>
    </figure> ➤

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_NN_tree_far_tree_base.jpg" alt="Stitched tree NN" class="img-33">
        <figcaption>Stitched tree (NN)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_tree_far_tree_base.jpg" alt="Stitched tree bilinear" class="img-33">
        <figcaption>Stitched tree (bilinear)</figcaption>
    </figure>
</div>
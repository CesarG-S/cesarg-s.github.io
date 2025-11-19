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
        <img src="/assets/images/proj3/playground_base.jpg" alt="Playground" class="img-33">
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/playground_far.jpg" alt="Playground turned" class="img-33">
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

With these points, we can solve for the elements of the homography matrix. The known values are the x's, the y's, the u's, and the v's, and the unknowns are the h's. I originally used the convention that points were (x,y) and (u,v), but I found it was much easier to use the convention that points were (y,x) and (v,u), since image arrays index with y as height and x as width. The ginput function still labels points (x,y) by default, so I had to swap the order before working with them later. Points with the same subscript are correspondences.

To make it as clear as possible, the homography matrix H is a matrix such that:
<div class="math-size-150">
    $$Hp = p'$$

    $$ \begin{bmatrix}  h_1 & h_2 & h_3 \\
                        h_4 & h_5 & h_6 \\
                        h_7 & h_8 & 1
    \end{bmatrix}
    \begin{bmatrix} y_1 \\ x_1 \\ 1 \end{bmatrix} 
    = 
    \begin{bmatrix} \lambda v_1 \\ \lambda u_1 \\ \lambda \end{bmatrix} $$
</div>

By expanding the matrix multiplication, we can extract three equations. Solving for lambda in the last equation, we can plug in lambda into the first two equations, and solve for v and u in terms of y, x, and h. The 1 is there as a result of homogeneous coordinates and the fact that we will always scale back down to 1. This means the h9 value is ultimately unimportant and we can just set it to 1.

To solve for the unknown h's, I set up a system of equations in the form:

<div class="math-size-150">
    $$Ah = b$$

    $$ \begin{bmatrix}  y_1 & x_1 & 1 & 0 & 0 & 0 & -y_1v_1 & -x_1v_1 \\
                        0 & 0 & 0 & y_1 & x_1 & 1 & -y_1u_1 & -x_1u_1 \\
                        y_2 & x_2 & 1 & 0 & 0 & 0 & -y_2v_2 & -x_2v_2 \\
                        0 & 0 & 0 & y_2 & x_2 & 1 & -y_2u_2 & -x_2u_2 \\
                        y_3 & x_3 & 1 & 0 & 0 & 0 & -y_3v_3 & -x_3v_3 \\    
                        0 & 0 & 0 & y_3 & x_3 & 1 & -y_3u_3 & -x_3u_3 \\
                        y_4 & x_4 & 1 & 0 & 0 & 0 & -y_4v_4 & -x_4v_4 \\
                        0 & 0 & 0 & y_4 & x_4 & 1 & -y_4u_4 & -x_4u_4
    \end{bmatrix}
    \begin{bmatrix} h_1 \\ h_2 \\ h_3 \\ h_4 \\ h_5 \\ h_6 \\ h_7 \\ h_8 \end{bmatrix} 
    = 
    \begin{bmatrix} v_1 \\ u_1 \\ v_2 \\ u_2 \\ v_3 \\ u_3 \\ v_4 \\ u_4 \end{bmatrix} $$
</div>

As you may have noticed, I have more than 4 correspondences. The system of equations follows the same pattern, but with more rows than unknowns. To solve this system of equations, I use least squares. This gives me the h's with which I may build my homography matrix H. As a final step, I append a 1 and reshape into a 3x3 matrix.

A crucial detail to understand in this calculation is in my case, given my function computeH(im1pts, im2pts), image 1 represents the "source" image, or where the pixels come from, and image 2 represents the "destination" image, or where the pixels will "warp to" after the calculations are made. This is an important distinction to make, because if you are trying to warp image 2 to look like image 1, then you need to find the correspondences and compute H using computeH(im2pts, im1pts). This little detail caused me a lot of headache while debugging.

With that in mind, using the above images im1 and im2 as inputs, and after manually selecting correspondences, we calculate the homography matrix:

```python
computeH(im1pts, im2pts)

# ========= Output ========= #
im1 points: # (x,y) values; must convert to (y,x) before using
 [
    [24, 379], [38, 379], [24, 397], [38, 397], 
    [79, 313], [149, 315], [78, 410], [147, 413], 
    [186, 343], [256, 343], [187, 440], [256, 441], 
    [299, 401], [326, 401], [299, 427], [326, 428], 
    [270, 565], [257, 576], [181, 494], [173, 556], 
    [124, 555], [121, 555], [210, 763], [210, 749]
 ]
im2 points: # (u,v) values; must convert to (v,u) before using
 [
    [410, 403], [421, 402], [410, 418], [421, 418], 
    [454, 339], [515, 332], [453, 427], [512, 426], 
    [550, 354], [622, 346], [550, 450], [621, 447], 
    [670, 402], [702, 399], [669, 430], [702, 429], 
    [638, 578], [623, 589], [544, 503], [537, 564], 
    [492, 560], [489, 560], [578, 775], [578, 761]
 ]
Homography matrix:
 [[ 8.35644865e-01 -3.45541747e-01  8.39557666e+01]
  [-7.28795876e-03  4.60968830e-01  3.91030864e+02]
  [-2.57185789e-05 -6.83015022e-04  1.00000000e+00]]
```

## Performing Image Warping and Registration
Now that we have our homography matrix, we can perform inverse mapping to rebuild a version of image 1 that matches the shape of image 2. That is, we will use the homography matrix to warp image 1 to look like image 2. Instead of applying H to every point from image 1, I start at a pixel in the warped image and use H inverse to figure out which pixel in the original image to sample. If we land outside the boundary of the original image 1, I ignore that pixel by setting its alpha value to 0. I talk more about alpha values later.

To find the edges of the warped image canvas, I take my original image, find the corner points, and matrix multiply with the homography matrix to see where they would land in the warped image canvas. I draw a bounding box around these points and keep track of where (0,0) lands with respect to the bounding box. The values of the top-left of the bounding box are very important when I later line up the warped image 1 and original image 2. I iterate over all points in the warped image canvas and create an augmented vector in the form (y, x, 1) (transposed) for use in inverse mapping. By matrix multiplying with H inverse, then scaling such that the last element is equal to 1, you get a set of points (v, u) to sample from the source image. Using np.meshgrid, I was able to vectorize this process to speed things up significantly. Essentially, I found all the points beforehand and performed inverse mapping in one go, filtering out pixels that are out of bounds.

The actual sampling of pixels can be done in many ways, but the two implementations I compared were nearest-neighbor interpolation and bilinear interpolation. I compare their performance later.

Now that we have our warped image, we need to line it up with image 2. The top-left corner of the bounding box from before defines the offset we use. If you imagine an infinite canvas and place the original image 2 in the center, then the warped image 1 should be placed in such a way where its origin matches image 2's origin. Warped image 1's origin is the top-left corner of the bounding box, so by offsetting warped image 1 such that the origins match, we will have properly lined up our images. In practice, I made a third canvas which could fit both images plus the offset, and shifted the warped image if the offset was negative and the original image if the offset was positive. The actual implementation is not too important, but the results are the same. As long as you can line up both origins, you will have lined up your images, given the homography calculations are correct.

Before we move on to warping and stitching actual images together, it is good to double check that the homography code works fine. I perform image rectification on images with rectangular objects of known dimensions. I took these photos at funny angles to make straightening them out more challenging. I recommend opening each image in a new tab to really see the difference.

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

Looks good. I just picked the rectangle nearest to 500x500 pixels with the same aspect ratio as the object in the photo, and mapped the image with the object to that rectangle. There is a clear visual improvement when using bilinear interpolation. The image looks a lot smoother, due to the fact that each pixel is a rough average of the surrounding pixels, while nearest-neighbor catpures the raw pixels, which may be of higher frequency. Using vectorization, bilinear interpolation ran about 3 times slower than nearest neighbor, but both ran in less than a second, so the difference doesn't really matter. Nearest-neighbor ran in roughly .15 seconds, versus .45 seconds for bilinear interpolation. I think this difference is small enough to make bilinear the clear winner.

## Image Stitching and Blending
Now for the real fun. To warp two images together, you manually select points that appear in both images and warp image 1 such that it fits to the correspondences in image 2. After warping and stacking, you are left with a panorama.

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

However, simply stacking the two images on top of each other produces a visible seam. To handle this, I created an alpha channel for each image before warping in the shape of a Gaussian centered at the middle of each image. After some experimentation, I picked sigma values of height/5 and width/5 for the y Gaussian and the x Gaussian, respectively. Perform an outer product, and normalize so that the center is 1, and you have a Gaussian alpha channel.

During the image stacking step, I found where image 1 and warped image 2 overlap, extracted the alpha channels, and performed alpha blending according to:

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
            <img src="/assets/images/proj3/gaussian_alpha_playground_base.jpg" alt="Playground base alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/playground_base.jpg" alt="Playground base" class="img-33 default-image">
        </div>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_playground_far.jpg" alt="Playground far alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/playground_far.jpg" alt="Living room far" class="img-33 default-image">
        </div>
    </figure> ➤

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_NN_playground_base_playground_far.jpg" alt="Stitched playground NN" class="img-33">
        <figcaption>Stitched playground (NN)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_playground_base_playground_far.jpg" alt="Stitched playground bilinear" class="img-33">
        <figcaption>Stitched playground (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_ducks_base.jpg" alt="Duck base alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/ducks_base.jpg" alt="Duck far" class="img-33 default-image">
        </div>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_ducks_far.jpg" alt="Duck far alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/ducks_far.jpg" alt="Duck far" class="img-33 default-image">
        </div>
    </figure> ➤

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_NN_ducks_base_ducks_far.jpg" alt="Stitched duck NN" class="img-33">
        <figcaption>Stitched ducks (NN)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_ducks_base_ducks_far.jpg" alt="Stitched duck bilinear" class="img-33">
        <figcaption>Stitched ducks (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_tree_base.jpg" alt="Tree base alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/tree_base.jpg" alt="Tree base" class="img-33 default-image">
        </div>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/gaussian_alpha_tree_far.jpg" alt="Tree far alpha channel" class="img-33 hover-image">
            <img src="/assets/images/proj3/tree_far.jpg" alt="Tree far" class="img-33 default-image">
        </div>
    </figure> ➤

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_NN_tree_base_tree_far.jpg" alt="Stitched tree NN" class="img-33">
        <figcaption>Stitched tree (NN)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_tree_base_tree_far.jpg" alt="Stitched tree bilinear" class="img-33">
        <figcaption>Stitched tree (bilinear)</figcaption>
    </figure>
</div>

## Automating the process with Harris Corner Detection
Going through and manually clicking points is extremely tedious. Not only do you have to zoom in and click points that appear in both images, you have to do it twice, and in the same order, and even then the second image is slightly different from the first, and it may not be obvious which pixel is the matching one. Hopefully that sentence was enough to convince you that I did not particularly enjoy manually selecting correspondences. Thanfully, there exist techniques to automatically identify points of interest in an image. I use the provided Harris Interest Point Detector and overlayed the resulting points over my image.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/harris_zoom_livingroom_base.jpg" alt="Harris living room base zoom" class="img-20 hover-image">
            <img src="/assets/images/proj3/harris_livingroom_base.jpg" alt="Harris living room base dense" class="img-20 default-image">
        </div>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj3/harris_zoom_livingroom_far.jpg" alt="Harris living room far zoom" class="img-20 hover-image">
            <img src="/assets/images/proj3/harris_livingroom_far.jpg" alt="Harris living room far dense" class="img-20 default-image">
        </div>
    </figure>
</div>

Oh, that's a lot of corners. But surely not every single one is good. A lot of those points land on the wall, which has no pattern, so those surely should not be used to solve for a homography. Also, there's a lot of points all huddled up next to each other. There will be issues with distinguishing points if they are allowed to be close together. Thankfully, ANMS solves both problems by keeping only the best corners in a neighborhood, or radius, of pixels.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/anms_livingroom_base.jpg" alt="ANMS living room base dense" class="img-20">
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/anms_livingroom_far.jpg" alt="ANMS living room far dense" class="img-20">
    </figure>
</div>

But these pixels alone cannot tell us much. We must do feature description for each corner. To do this, take the 40x40 patch centered around the corner, apply a blur, and downsize to 8x8. From there, normalize the image such that it has mean 0 and standard deviation 1. Remember, images are just rows and columns with numbers, so it's easy to find the mean and standard deviation of the image as a whole.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/patches_livingroom_base.jpeg" alt="Patches living room base" class="img-50">
        <figcaption>Patches for image 1</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/patches_livingroom_far.jpeg" alt="Patches living room far" class="img-50">
        <figcaption>Patches for image 2</figcaption>
    </figure>
</div>

Rather luckily, we can see that patch 2 in both images is a correspondence. But to actually figure that out, we must go through each patch in image 1, and compare with each patch in image 2. I use the L2 difference to compare patches, and then I take the best two matches and do a test. If the best patch is a much better match than the second-best patch, then I will declare that that patch in image 1 matches with the best-matching patch in image 2, and save the corresponding points. If the best patch and the second-best patch are fairly similar in terms of how much they match the patch in image 1, then I pick neither, and keep going. The threshold I use is 1-NN/2-NN < 0.6 to keep matches. This keeps a good portion of the correct matches while retaining only a few incorrect matches. 

Now that I have this set of mostly good and only a few bad matches, I use RANSAC to try to find an estimate to the points whose homography matches the most points in images 1 and 2. The idea is that with the correct homography matrix, the corners we found in image 1 will map to a corresponding corner in image 2 after applying a homography transformation. At this point, however, we don't know the best homography matrix, so we pick random points and try a bunch of different exact homographies (only 4 points used to build H) and see how well they do in terms of properly assigning corners in image 1 to corners in image 2. The random homography that produces the largest set of inliers is the winner. We save the largest set of inliers to perform least squares to find a really good estimate for H.

Below is a visualization of the entire process. The red dots are points located by ANMS, blue dots are points with a corresponding strong match after matching patches, and green dots are the best set of inliers.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/correspondences_livingroom_base_livingroom_far.jpeg" alt="All matches living room" class="img-67">
        <figcaption>Living room corner overlay</figcaption>
    </figure>
</div>

All together, this process saves me a lot of time, since I no longer have to manually find correspondences. Check out the final results and compare with manually picking correspondences:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/automatic_match_bilinear_livingroom_base_livingroom_far.jpg" alt="Automatic living room" class="img-50">
        <figcaption>Living room automatic match (bilinear)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_livingroom_base_livingroom_far.jpg" alt="Manual living room" class="img-50">
        <figcaption>Living room manual match (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/automatic_match_bilinear_playground_base_playground_far.jpg" alt="Automatic playground" class="img-50">
        <figcaption>Playground automatic match (bilinear)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_playground_base_playground_far.jpg" alt="Manual playground" class="img-50">
        <figcaption>Playground manual match (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/automatic_match_bilinear_ducks_base_ducks_far.jpg" alt="Automatic ducks" class="img-50">
        <figcaption>Ducks automatic match (bilinear)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_ducks_base_ducks_far.jpg" alt="Manual ducks" class="img-50">
        <figcaption>Ducks manual match (bilinear)</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/automatic_match_bilinear_tree_base_tree_far.jpg" alt="Automatic tree" class="img-50">
        <figcaption>Tree automatic match (bilinear)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj3/stitched_blended_bilinear_tree_base_tree_far.jpg" alt="Manual tree" class="img-50">
        <figcaption>Tree manual match (bilinear)</figcaption>
    </figure>
</div>

I'm super satisfied with this work. The automatic stitcher certainly got more points than me and was able to more accurately pick matches.
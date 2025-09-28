---
layout: default
permalink: /project2/
title: CS180 - Project 2
---
## Introduction
The goal of this project is to play around with images. 

## Fun with Filters: Implementing Convolutions
Convolutions have two parts - your image and your kernel. The kernel is a rectangle that slides over your image and applies an element-wise multiplication with the pixels that overlap. The first step before you perform this shifting and multiplication procedure is to flip the kernel horizontally and vertically. This was pretty easy with np.flip. We also want the output matrix to be the same size as our original matrix, so we need to pad our image with zeroes so that the middle of the kernel starts at the top-left pixel of our image and ends at the bottom-right pixel. There are other ways to pad your image, but I proceed with zero-padding.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/convolution.gif" alt="Graphic of a convolution" class="img-33">
        <figcaption>An example convolution. Notably, it is not padded.</figcaption>
    </figure>
</div>

The naive implementation involves four for loops. You need two loops to iterate over every pixel of the image, and two more loops for each pixel of the kernel. In the two innermost for loops, you perform element-wise multiplication with the current kernel pixel and the image pixel that overlaps with the kernel pixel. You sum up the products from each kernel pixel and store that sum in the final output matrix before shifting to the next pixel and recalculating the sum.

```python
def manual_convolve_fourloop(image, kernel):
    height, width = image.shape

    # Flip kernel horizontally and vertically
    kernel = np.flip(kernel, axis=(0,1))
    k_height, k_width = kernel.shape

    # Zero-pad image
    image = np.pad(image, [(k_height//2, k_height//2), (k_width//2, k_width//2)], mode='constant', constant_values=0)

    output = np.zeros((height, width))

    # Slide the kernel over the image
    for x in range(height):
        for y in range(width):
            total = 0
            # For each element of the kernel, go one by one and multiply element-wise the image and kernel
            for kx in range(k_height):
                for ky in range(k_width):
                    total += image[x+kx, y+ky] * kernel[kx, ky]
            output[x, y] = total
    
    return output
```

<br>
You can eliminate the two innermost for loops by directly multiplying the patch of image that overlaps with the kernel with the entire kernel. You can do this by slicing the image matrix such that it is the same shape as the kernel. From there, you perform element-wise multiplication and sum all the products.

```python
def manual_convolve_twoloop(image, kernel):
    height, width = image.shape

    # Flip kernel horizontally and vertically
    kernel = np.flip(kernel, axis=(0,1))
    k_height, k_width = kernel.shape

    # Zero-pad image
    image = np.pad(image, [(k_height//2, k_height//2), (k_width//2, k_width//2)], mode='constant', constant_values=0)

    output = np.zeros((height, width))

    # Slide the kernel over the image
    for x in range(height):
        for y in range(width):
            # Directly take the patch of the image aligned with the kernel and multiply element-wise
            total = np.sum(image[x:x+k_height, y:y+k_width] * kernel)
            output[x, y] = total
    
    return output
```

Below is an image of myself I took a while ago, in black-and-white. The size of the image is 2268x3024 pixels.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/bw_selfie.jpg" alt="Black-and-white selfie" class="img-33">
        <figcaption>Black-and-white selfie</figcaption>
    </figure>
</div>

The first convolution I will apply is one with a 9x9 box filter. The output image will be the result of taking every pixel from the original image, finding all the neighbors within a 9x9 square centered at that pixel, and averaging the values. So for example, the middle pixel of the output will come from taking the middle pixel in the original and averaging with the values of its neighbors in a 9x9 box.

$$ \text{box} = \frac{1}{81} \begin{bmatrix} 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \\ 1, 1, 1, 1, 1, 1, 1, 1, 1 \end{bmatrix} $$

The effect is the image will be blurred. Hover over each image below to see the box filter be applied. It might be hard to see unless you zoom in very closely.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv4loop_box_selfie.jpg" alt="Box filter using the naive 4 loop convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Box filter with 4 loop convolution (205.65 sec)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv2loop_box_selfie.jpg" alt="Box filter using the improved 2 loop convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Box filter with 2 loop convolution (38.024 sec)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv2loop_box_selfie.jpg" alt="Box filter using scipy convolve2d convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Box filter with scipy convolve2d convolution (1.26 sec)</figcaption>
    </figure>
</div>

The next convolution I apply is the finite difference in the x direction. This kernel acts as a vertical edge detector, as changes in pixel value in the x direction (across a vertical line) will appear very bright.

<div class="math-size-150">
    $$ D_x = \begin{bmatrix} 1, 0, -1 \end{bmatrix} $$
</div>

This time, the effect is very pronounced, and maybe even a bit scary. Hover over each image to see the result of applying the Dx kernel.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv4loop_Dx_selfie.jpg" alt="Dx using the naive 4 loop convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dx with 4 loop convolution (11.9 sec; interesting!)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv2loop_Dx_selfie.jpg" alt="Dx using the improved 2 loop convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dx with 2 loop convolution (38.96 sec)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv2loop_Dx_selfie.jpg" alt="Dx using scipy convolve2d convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dx with scipy convolve2d convolution (0.183 sec)</figcaption>
    </figure>
</div>

For the next convolution, I do the same, but in the y direction. This time, the kernel is a horizontal edge detector, for similar reasons as the Dx kernel. Changes in the y direction (across horizontal rows, going down) will appear very bright; you should see a bright line where a horizontal edge exists.

<div class="math-size-150">
  $$ D_y = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} $$
</div>


The effect is similar, but for horizontal edges. Hover over each image to see the result of applying the Dy kernel.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv4loop_Dy_selfie.jpg" alt="Dy using the naive 4 loop convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dy with 4 loop convolution (14.33 sec)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv2loop_Dy_selfie.jpg" alt="Dy using the improved 2 loop convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dy with 2 loop convolution (38.3 sec)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/conv2loop_Dy_selfie.jpg" alt="Dy using scipy convolve2d convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dy with scipy convolve2d convolution (0.289 sec)</figcaption>
    </figure>
</div>

If you look closely, however, you should be able to tell that something is wrong. The way the images are displayed, black pixels represent 0 and white pixels represent positive values. However, there should exist negative values. Probably where this would be most apparent is the boundary between the wall and my shoulder, where a hard horizontal line exists. However, this line doesn't appear in the image due to the negative values being squished to 0. Applying a processing step after calculating the convolutions such that the middle value is 128, instead of 0, results in these better images:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/convscipy_Dx_normalized_selfie.jpg" alt="Dx using normalized scipy convolve2d convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dx with visible negative values</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/convscipy_Dy_normalized_selfie.jpg" alt="Dy using normalized scipy convolve2d convolution" class="img-33 hover-image">
            <img src="/assets/images/proj2/bw_selfie.jpg" alt="Original black-and-white selfie" class="img-33 default-image">
        </div>
        <figcaption>Dy with visible negative values</figcaption>
    </figure>
</div>

Now it is very obvious how the Dx and Dy kernels act as vertical and horizontal edge detectors, respectively. 

I also want to note that the two-loop convolution is not strictly faster than the four-loop convolution function. It turns out the overhead for creating numpy slices ends up mattering a lot, and essentially adds a constant factor of time to the runtime, whereas the four-loop version directly indexes into the image and kernel. Of course, this is where the four-loop version's advantage ends, as this victory only matters for very small kernels. For larger kernels, the vectorization in the numpy sum and multiply operations is much more important in saving time. For the 9x9 box filter, the four-loop version took about 5 times longer to run than the two-loop version, but was more than 3 times faster for the small finite difference filters. The scipy convolve2d() function demolishes both my implementations in runtime, performing 200 times faster than my four-loop version in the box filter test and about 30 times faster than the two-loop version. However, for the finite difference tests, convolve2d() was 50-65 times faster than the four-loop version, and over 130-200 times faster than the two-loop version.

## Gradient Magnitude
The next image I will play with is the cameraman image below.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/cameraman.png" alt="Black-and-white cameraman image" class="img-33">
        <figcaption>cameraman.png</figcaption>
    </figure>
</div>

From left to right, these images represent the output of convolving with the Dx kernel, the Dy kernel, and computing the resulting gradient magnitude matrix. To compute the gradient magnitude, I took the square root of the sum of Dx^2 and Dy^2 element-wise for each pixel. The Dx and Dy images are shifted up so that the middle value is 128.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/Dx_normalized_cameraman.png" alt="Dx normalized on cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>Dx with visible negative values</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/Dy_normalized_cameraman.png" alt="Dy normalized on cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>Dy with visible negative values</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/gradient_magnitude_cameraman.png" alt="Gradient magnitude of cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>Gradient magnitude</figcaption>
    </figure>
</div>

I went ahead and binarized the gradient magnitude image, meaning below some pixel value threshold, all pixels are set to 0, and above the threshold, all pixel values are set to 1. I found 0.28 to be a good threshold for my images, though 0.35 removed essentially all the noise, at the cost of a few edge pixels.

<div class="image-row"> 
    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/gradient_magnitude_binarize10_cameraman.png" alt="Gradient magnitude with binarize threshold 0.10" class="img-33">
        <figcaption>Threshold 0.10</figcaption>
    </figure>

    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/gradient_magnitude_binarize20_cameraman.png" alt="Gradient magnitude with binarize threshold 0.20" class="img-33">
        <figcaption>Threshold 0.20</figcaption>
    </figure>

    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/gradient_magnitude_binarize28_cameraman.png" alt="Gradient magnitude with binarize threshold 0.28" class="img-33">
        <figcaption>Threshold 0.28</figcaption>
    </figure>

    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/gradient_magnitude_binarize35_cameraman.png" alt="Gradient magnitude with binarize threshold 0.35" class="img-33">
        <figcaption>Threshold 0.35</figcaption>
    </figure>
</div>

## Combining Convolutions
There was a good amount of noise causing issues in my images, but by applying a Gaussian kernel convolution over the image first, we can minimize the noise (by blurring the image) and hopefully have a cleaner gradient magnitude image. I use a 9x9 Gaussian kernel using cv2.getGaussianKernel().

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/blur_cameraman.png" alt="Blur cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>Blurred cameraman.png</figcaption>
    </figure>
</div>

In the following images, I apply a Gaussian kernel convolution to the cameraman image before applying either Dx or Dy.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/blur_Dx_normalized_cameraman.png" alt="Blur Dx normalized on cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>Blur+Dx with visible negative values</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/blur_Dy_normalized_cameraman.png" alt="Blur Dy normalized on cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>Blur+Dy with visible negative values</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/blur_gradient_magnitude_cameraman.png" alt="Blur gradient magnitude of cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>Blur gradient magnitude</figcaption>
    </figure>
</div>

However, convolutions are commutative, meaning you can apply the kernel convolutions on each other before finally applying a single convolution on the cameraman image.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/dog_x_normalized_cameraman.png" alt="DoG x normalized on cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>DoG in x with visible negative values</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/dog_y_normalized_cameraman.png" alt="DoG y normalized on cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>DoG in y with visible negative values</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/dog_gradient_magnitude_cameraman.png" alt="DoG gradient magnitude of cameraman" class="img-33 hover-image">
            <img src="/assets/images/proj2/cameraman.png" alt="Cameraman" class="img-33 default-image">
        </div>
        <figcaption>DoG gradient magnitude</figcaption>
    </figure>
</div>

The threshold values must be lowered to not lose the cameraman's edges. I liked the threshold of 0.15.

<div class="image-row"> 
    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/dog_gradient_magnitude_binarize15_cameraman.png" alt="DoG gradient magnitude with binarize threshold 0.15" class="img-33">
        <figcaption>Threshold 0.15</figcaption>
    </figure>

    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/dog_gradient_magnitude_binarize20_cameraman.png" alt="DoG gradient magnitude with binarize threshold 0.20" class="img-33">
        <figcaption>Threshold 0.20</figcaption>
    </figure>

    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/dog_gradient_magnitude_binarize28_cameraman.png" alt="DoG gradient magnitude with binarize threshold 0.28" class="img-33">
        <figcaption>Threshold 0.28</figcaption>
    </figure>
</div>

These were the filters that were created as a result of combining all convolutions before applying them. It's important to note that the values are centered at 0.5 for easier visualization.

<div class="image-row"> 
    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/dog_x_filter.png" alt="DoG x filter" class="img-33">
        <figcaption>DoG in x filter</figcaption>
    </figure>

    <figure class="image-with-subtitle">
            <img src="/assets/images/proj2/dog_y_filter.png" alt="DoG y filter" class="img-33">
        <figcaption>DoG in y filter</figcaption>
    </figure>
</div>

I think the effect of applying a Gaussian blur to the cameraman image when calculating the gradient magnitude is rather significant. The edge lines are thicker and there is significantly less noise. The intensity of the edge lines are less in the blurred version, so a lower threshold is needed, but it appears much cleaner, even if some noise is still present.

## Fun with Frequencies: Image Sharpening
If a Gaussian blue reduces high frequencies, then to get those high frequencies back, it makes sense to subtract the original image with the blurred image. We can take this idea to "sharpen" an image - by adding an image's high frequencies to the original, you get a sharper image. 

The procedure is as follows: convolve the original image with a Gaussian kernel, take the original image and subtract it with the blurred image, then add the original image to this image of high frequencies.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/taj.jpg" alt="Taj" class="img-33">
        <figcaption>Original</figcaption>
    </figure>

    -

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/blur_taj.jpg" alt="Taj blur" class="img-33">
        <figcaption>Blur</figcaption>
    </figure>

    =

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/highlights_taj.jpg" alt="Taj highlights" class="img-33">
        <figcaption>Highlights</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/taj.jpg" alt="Taj" class="img-33">
        <figcaption>Original</figcaption>
    </figure>

    +

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/highlights_taj.jpg" alt="Taj highlights" class="img-33">
        <figcaption>Highlights</figcaption>
    </figure>

    =

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/sharp_taj.jpg" alt="Taj sharp" class="img-33">
        <figcaption>Sharpened</figcaption>
    </figure>
</div>

This is fine, but we could do it directly in one convolution if there existed a kernel such that convolving an image with the kernel resulted in an identical image. Such a filter does exist - you take a square with all zeroes except for a 1 at the center coordinates. I call this matrix my impulse matrix.

My unsharp matrix is then defined as:
<br>
Impulse + amount * (Impulse - Gaussian)

The amount parameter dictates how many times to perform sharpening.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/unsharp_mask_taj.jpg" alt="Taj 1x sharp" class="img-33">
        <figcaption>Sharpened amount=1</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/unsharp_2_mask_taj.jpg" alt="Taj highlights" class="img-33">
        <figcaption>Sharpened amount=2</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/unsharp_4_mask_taj.jpg" alt="Taj sharp" class="img-33">
        <figcaption>Sharpened amount=4</figcaption>
    </figure>
</div>

Convolving my images with this unsharp mask results in a "sharper" image:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/unsharp_mask_mark.png" alt="Sharpened mark" class="img-33 hover-image">
            <img src="/assets/images/proj2/mark.png" alt="Original mark" class="img-33 default-image">
        </div>
        <figcaption>mark.png</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/unsharp_mask_lincoln.jpg" alt="Sharpened lincoln" class="img-33 hover-image">
            <img src="/assets/images/proj2/lincoln.jpg" alt="Original lincoln" class="img-33 default-image">
        </div>
        <figcaption>lincoln.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/unsharp_mask_bigfoot.jpg" alt="Sharpened bigfoot" class="img-33 hover-image">
            <img src="/assets/images/proj2/bigfoot.jpg" alt="Original bigfoot" class="img-33 default-image">
        </div>
        <figcaption>bigfoot.jpg</figcaption>
    </figure>
</div>

Here, I take a sharp image, then blur it, then sharpen it again to compare how good the unsharp mask is. Notably, it is hard to get back the highlights of an image after you have blurred it. The unsharp filter is not magic; there has to be highlights to enhance or else the sharpening doesn't really do much.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/frog.jpg" alt="Frog" class="img-33">
        <figcaption>frog.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/sharp_frog.jpg" alt="Frog sharp" class="img-33">
        <figcaption>Sharpened frog.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/sharp_blur_frog.jpg" alt="Blur then sharp frog" class="img-33">
        <figcaption>Blurred frog.jpg, then sharpened</figcaption>
    </figure>
</div>



## Hybrid Images
By combining the low frequencies of one image with the high frequencies of another in a certain way, you can trick yourself into seeing two different images in one. When you stand far away, you see something different than when you look from up close. 

The process is not too difficult to understand. You capture an image's low frequencies with a Gaussian blur and another image's high frequencies by subtracting a Gaussian blurred image from the original. You then add the two resulting low and high frequency images. However, this process is not an exact science, and extra parameters need introduction.

The parameters I played with were the sigma values for each Gaussian, the amount to boost the low and high frequencies in the hybrid image, and the presence of either original image mixed with the hybrid. Strictly speaking, only the first parameter is truly important for the effect to actually work, but the other two are minor tweaks which help make an image that is visually pleasing. For the sigmas, a higher sigma roughly corresponds to a stronger blur, meaning if used to capture low frequencies, a high sigma would produce a very blurry image. If used to capture high frequencies, a high sigma would produce a very sharp image. I automatically assigned the kernel size to be equal to 6*sigma + 1. For the boosts, I could choose to strengthen the high frequencies and diminish the low frequencies, or vice versa, or strengthen both or diminish both. For incorporating the original images, I just took a fraction of either original image and added it directly to the hybrid, if needed.

For Derek and Nutmeg, the sigmas I chose were 10.0 for the low frequencies, 5.0 for the high frequencies. To be completely honest, this process was heavily dependent on trial and error. This was simply the combination that I liked best. When boosting either the high or low frequencies, I boosted the highs by multiplying by 1.2 and left the lows as is. In terms of readding an original image, I added 0.2 times the image of Nutmeg and did not add any more of Derek.


From start to finish, the process is as follows: align your input images. 
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/derek.jpg" alt="Derek" class="img-33">
        <figcaption>derek.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/nutmeg.jpg" alt="Nutmeg" class="img-33">
        <figcaption>nutmeg.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj2/aligned_derek.jpg" alt="Derek aligned" class="img-33 hover-image">
            <img src="/assets/images/proj2/aligned_nutmeg.jpg" alt="Nutmeg" class="img-33 default-image">
        </div>
        <figcaption>derek.jpg aligned over nutmeg.jpg</figcaption>
    </figure>
</div>

Apply a low-pass filter on the first and a high-pass filter on the second, and then stack them on top of each other. Tweak parameters as needed until final image looks good. 
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/filtered_derek.jpg" alt="Derek low-pass" class="img-20">
        <figcaption>derek.jpg after low-pass filter</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/filtered_nutmeg.jpg" alt="Nutmeg high-pass" class="img-20">
        <figcaption>nutmeg.jpg after high-pass filter</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/hybrid_derek_nutmeg.jpg" alt="Hybrid Derek+Nutmeg" class="img-20">
        <figcaption>Final hybrid image</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/fft_derek.jpg" alt="FFT Derek" class="img-33">
        <figcaption>FFT of Derek (before alignment)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/fft_nutmeg.jpg" alt="FFT Nutmeg" class="img-33">
        <figcaption>FFT of Nutmeg (before alignment)</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/fft_low_derek.jpg" alt="FFT Low Derek" class="img-33">
        <figcaption>FFT of Derek low frequencies + alignment</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/fft_high_nutmeg.jpg" alt="FFT High Nutmeg" class="img-33">
        <figcaption>FFT of Nutmeg high frequencies + alignment</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/fft_hybrid_derek_nutmeg.jpg" alt="FFT Hybrid Derek+Nutmeg" class="img-33">
        <figcaption>FFT of Hybrid Derek+Nutmeg</figcaption>
    </figure>
</div>

I made more hybrid images and manually cropped them. I'm really proud of the Obama one and the JFK one, but the Einstein one is admittedly kinda creepy (the effect still works, though):

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/derek.jpg" alt="Derek" class="img-20">
        <figcaption>derek.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/nutmeg.jpg" alt="Nutmeg" class="img-20">
        <figcaption>nutmeg.jpg</figcaption>
    </figure>
    
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/hybrid_derek_nutmeg_cropped.jpg" alt="Hybrid Derek+Nutmeg" class="img-20">
        <figcaption>Hybrid of Derek+Nutmeg</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/einstein.jpg" alt="Einstein" class="img-20">
        <figcaption>einstein.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/bichon.jpg" alt="Bichon" class="img-20">
        <figcaption>bichon.jpg</figcaption>
    </figure>
    
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/hybrid_einstein_bichon_cropped.jpg" alt="Hybrid of Bichon+Einstein" class="img-20">
        <figcaption>Hybrid of Einstein+Bichon</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/obama.jpg" alt="Obama" class="img-20">
        <figcaption>obama.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/obama_stern.jpg" alt="Obama stern" class="img-20">
        <figcaption>obama_stern.jpg</figcaption>
    </figure>
    
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/hybrid_obama_obama_stern_cropped.jpg" alt="Hybrid Obama+Obama stern" class="img-20">
        <figcaption>Hybrid of Obama+Obama stern</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/jfk.jpg" alt="JFK" class="img-20">
        <figcaption>jfk.jpg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/reagan.jpg" alt="Reagan" class="img-20">
        <figcaption>reagan.jpg</figcaption>
    </figure>
    
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/hybrid_jfk_reagan_cropped.jpg" alt="Hybrid JFK+Reagan" class="img-20">
        <figcaption>Hybrid of JFK+Reagan</figcaption>
    </figure>
</div>

## Blending Images
The idea of combining high and low frequencies to create hybrid images makes for a cool visual illusion, but we can use the idea of combining frequencies to nicely blend two images together. Simply taking the left half of an image and combining with the right half works for very simple repetitive patterns, but more likely than not, it just looks too jarring. We can try blurring the ends of each half when combining, but depending on how strong and where you blend, you may encounter harsh visual seams or ghosting, where the images clearly overlap with each other. 

To avoid seams, we want the window to equal the size of the largest prominent feature, but to avoid ghosting, we want the window to be at most twice the size of the smallest prominent feature.

This isn't always possible, but if we separate our image into frequency bands such that within our band, the largest frequency is at most twice the size of the smallest frequency, then we can easily find a window size that satisfies both conditions.

The idea is to separate our image into a Laplacian stack; each level of the stack is created by subtracting two low-pass filtered images so that the result only contains the frequencies contained by the first that are not in the second. By creating a Gaussian stack first, you can easily iterate over each pair of images to create the Laplacian stack. The final image in the Laplacian stack has to be the last image of your Gaussian stack because the subtraction step needs something to work with at the end. The idea is you can essentially work backwards if needed to create your Gaussian stack from your Laplacian, by adding the last two items of the Laplacian and working backwards to build up your Gaussian stack. With these final low frequencies, it is possible to rebuild the original image by adding all images of the Laplacian stack, which wouldn't be possible without that last step of adding the last element of the Gaussian stack. To create the Gaussian stack, simply apply a Gaussian blur to your starting image, then create a new image by blurring the blurred image, and so on as many times as desired.

At the same time, create a Gaussian stack for the mask you wish to use to blend your two images. The full process is as follows:

Take your starting images:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/apple.jpeg" alt="Apple" class="img-20">
        <figcaption>apple.jpeg</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/orange.jpeg" alt="Orange" class="img-20">
        <figcaption>orange.jpeg</figcaption>
    </figure>
    
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/mask_apple_orange.jpeg" alt="Mask" class="img-20">
        <figcaption>Mask</figcaption>
    </figure>
</div>

Create Gaussian and Laplacian stacks for your two input images. Creating only the Gaussian stack for the mask is sufficient. It is important to note that in my process, I create an extra Gaussian image in the stack for my mask, since the first item of my Gaussian stack is just the original image. The way I do this while keeping the behavior the same for the apple and orange is I create the desired amount of Gaussian stack images plus 1. I create the Laplacian stack as usual, excluding the final image, so the final image of the Laplacian stack is the second-to-last image of the Gaussian stack. Then, when I return my Gaussian stack, I return all but the first item, which essentially contains all the blurred versions of the original image, excluding the original itself. This doesn't matter for the input images, because the Gaussian stack is discarded after it is created, but it matters for the mask. The reason I do this is because if I collapsed my Laplacian stacks with the masks, the first step would have a hard 1 to 0 boundary, which looked obvious in my final image. By offsetting the masks by 1, my blended image starts blending the first two images with a blurred mask right off the bat.

An additional note is the Laplacian stack images are visualized in such a way such that the minimum value pixel is 0 and the maximum value pixel is 1. The range of the Laplacian images tends to be in a small range spanning some negative and positive float values, and plotting these was hard to see. By plotting with the minimum and maximums as the basis for 0 and 1, we can better see what each layer roughly contains. My actual calculations still use the real Laplacian values, of course.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/apple_stacks.png" alt="Apple stacks" class="img-50">
        <figcaption>Gaussian and Laplacian stacks for apple <br> Each image in the bottom row was created by taking the image in the row above and subtracting with the image in the row above to the right. The last image of the bottom row is just the last image of the top row.</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/orange_stacks.png" alt="Orange stacks" class="img-50">
        <figcaption>Gaussian and Laplacian stacks for orange</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/mask_stack.png" alt="Mask stacks" class="img-50">
        <figcaption>Gaussian stack for mask (note the extra value). The first mask is skipped when blending the two images.</figcaption>
    </figure>
</div>

The final image was created by multiplying the first level of the apple Laplacian stack with the second level of the mask Gaussian stack and adding with the first level of the orange Laplacian stack with 1 minus the second level of the mask Gaussian stack, then repeating the process for every level of the Laplacian stacks. Assuming you remove the first image of the mask Gaussian stack as I did, you can simplify to:

<div class="math-size-150">
    $$ l_i = m_i*l_i^A + (1-m_i)*l_i^B $$
</div>

where l_i is the blended image of level i of the Laplacian stack, l^A is the Laplacian stack for your first image, l^B is the Laplacian stack for your second image, and m is your mask Gaussian stack.

Then, to get your blended image, you simply sum up all the l_i images.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/blended_apple_orange.jpeg" alt="Oraple" class="img-20">
        <figcaption>Oraple</figcaption>
    </figure>
</div>

I repeated the process for a few more images. 
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/blended_lemon_eye.jpg" alt="Lemon eye" class="img-20">
        <figcaption>Leyemon</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/blended_pizza_cookie.jpg" alt="Pizza cookie" class="img-20">
        <figcaption>Pizookiea</figcaption>
    </figure>

    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/blended_watermelon_melon.jpg" alt="Watermelon melon" class="img-20">
        <figcaption>Watermelonmelon</figcaption>
    </figure>
</div>

The lemon was inspired by the Omega Mart lemon. If you don't know what that is, I recommend you look it up. Below is the mask I used to create the lemon image:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj2/eye_lemon_mask.jpg" alt="Lemon eye mask" class="img-20">
        <figcaption>Lemon eye mask</figcaption>
    </figure>
</div>


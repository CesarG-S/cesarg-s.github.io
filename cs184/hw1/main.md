---
layout: default
permalink: /cs184/hw1/
title: CS184 - Homework 1
---
Cesar Garcia Santana, Ethan Ye
## Overview
In this project, we learned to apply basic rasterization techniques in order to open simple SVG files. We approached different ways to store intermediate and final pixel data and colors, applied transformations to shapes, and mapped textures, with and without antialiasing. It was interesting to learn through repeated testing the importance of speed and optimization, since doing things through brute force would just make things unusably slow.

## Task 1: Rasterizing Triangles
To rasterize a triangle, you must first come up with a way to determine whether or not a pixel is inside of a triangle. A triangle is made up of three lines that enclose an area, so what counts as inside? There is a formula that allows you to check whether or not a point lies to the left/inside or right/outside of a line/triangle based on whether the result is positive or negative.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/linetest_formula.png" alt="Formula for left/right of line" class="img-50">
        <figcaption>Line test formula</figcaption>
    </figure>
</div>

If you walk counterclockwise along the triangle, and all the lines agreed that the point was on the "left," then you could safely conclude that the point lies within the triangle. It actually turns out that as long as you walk along the edge of the triangle, if all lines agree on which side of the line the point lies on, then the point must be in the triangle.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/triangle_test.png" alt="A depiction of the triangle left/right test" class="img-33">
        <figcaption>Line test for triangles</figcaption>
    </figure>
</div>

Given the coordinates of the three corners of the triangle, we can figure out the formulas for the lines that connect them and perform the direction check. In code, we compare the pixel center (offset by + 0.5) to the line, instead of the pixel corner (no offset). However, if the triangle is in the top-right of the canvas, then we don't want to check the bottom-left pixels, since that would be useless and hurt performance. Instead, we simply find the bounding box the triangle lives inside of and perform the checks on the pixels within that region. This dramatically reduces the area that we must check and improves performance by a lot.

If we find that a pixel passes the triangle test, then we can color in that pixel. To do that, we use a sample buffer to store intermediate color values per pixel, and then copy them over to the frame buffer. The reason for why we don't copy directly to the frame buffer is to support supersampling.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/task1.png" alt="A canvas showing rasterized triangles" class="img-33">
        <figcaption>Triangle rasterization (test4.svg)</figcaption>
    </figure>
</div>

## Task 2: Supersampling
Supersampling involves taking multiple subpixel samples per pixel and averaging the results. Supersampling is useful because it reduces jaggies and can reduce rendering artifacts like aliasing, at the cost of performance. Behind the scenes, each pixel holds multiple subpixels which must all be stored in memory. In our code, this meant resizing RasterImp::sample_buffer to hold height * width * sample_rate elements. We stored subpixels contiguously rather than in their "correct" 2D location, since it simplifies loops and has caching benefits.

For points, supersampling just involves coloring all subpixels the same color for a given pixel coordinate. For lines, the same is true, but along all points that fall on the line.

For triangles, one must first find the largest bounding box that the triangle fits in and run through each pixel. However, this time, instead of checking only the pixel center for the triangle test, we split the pixel into subpixels and perform the triangle test with each of those subpixel centers. For a sampling rate of 4 (meaning 4 subpixels), we split each pixel into quadrants and check each of the quadrant centers. If the sampling rate was 9, then we split each pixel into 9 subpixels, and so on. Now when we color in a pixel, we color in the corresponding subpixel value in the sample buffer. In a sense, the sample buffer stores a higher resolution version of the final frame that we draw.

To go from this high resolution buffer to the proper resolution image, we must do some aggregation. This is as simple as summing up the colors at each subpixel per pixel, averaging them, and then sending that averaged color to its place in the frame buffer.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/task2_super1.png" alt="Rasterized triangles at sampling rate 1" class="img-50">
        <figcaption>Triangle supersampling; sampling rate = 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/task2_super4.png" alt="Rasterized triangles at sampling rate 4" class="img-50">
        <figcaption>Triangle supersampling; sampling rate = 4</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/task2_super16.png" alt="Rasterized triangles at sampling rate 16" class="img-50">
        <figcaption>Triangle supersampling; sampling rate = 16</figcaption>
    </figure>
</div>

## Task 3: Transforms
Transforms are an important part of a rendering engine, since they make simple the manipulation of many moving parts at once. Here we have a simple cubeman jumping in joy, looking up in the air. Or, depending on how you look at it, strapped to a medieval torture device.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/my_robot.png" alt="Robot svg tied" class="img-50">
        <figcaption>my_robot.svg</figcaption>
    </figure>
</div>

## Task 4: Barycentric Coordinates
Barycentric coordinates are essentially x-y coordinates but for triangles. Using a modified version of the line formulas in task 1, given a coordinate point, we can derive a value proportional to the distance away from a given line. For a triangle, if we calculate the proportional distance between the line and the opposite triangle corner, we can use that as a weight to normalize our distance formula such that the proportional distance from the line to the corner is 1. 

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/pq_distance.png" alt="Figure showing proportional distance from line" class="img-33">
    </figure>
</div>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/pq_formula.png" alt="Formula for proportional distance from line" class="img-50">
        <figcaption>Proportional distance formula</figcaption>
    </figure>
</div>

If our distance formula is L(x, y), then we define alpha to be L(x,y) / L(x_corner, y_corner), where x_corner and y_corner are the coordinate points of the triangle corner directly opposite to the line. We use a similar procedure on a different corner to construct beta. Lastly, we define gamma by 1 - alpha - beta.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/pq_alpha.png" alt="Calculation for alpha" class="img-20">
        <figcaption>Alpha equation</figcaption>
    </figure>
</div>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/barycentric_alpha.png" alt="Alpha for barycentric coordinates" class="img-20">
        <figcaption>Alpha representation for barycentric coordinates</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/barycentric.png" alt="Barycentric coordinate equations" class="img-33">
        <figcaption>Barycentric coordinate equations</figcaption>
    </figure>
</div>

The result is a coordinate system with many useful properties. If alpha, beta, and gamma are all positive for a given (x, y) point, then the point lies within the triangle. The values of alpha, beta, and gamma also represent the proportional area of each inner triangle formed when connecting each vertex to the (x, y) point within the triangle. More useful to us right now is the ability to linearly interpolate values within the triangle. 

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/barycentric_area.png" alt="Barycentric coordinates proportional area depiction" class="img-33">
        <figcaption>Proportional areas for barycentric coordinates</figcaption>
    </figure>
</div>

Since alpha, beta, and gamma correspond to a proportional weight based on the location of the point within the triangle, if we let each triangle vertex represent a color, then we can obtain intermediate colors by multiplying the color of each vertex by its corresponding weight and summing together the result. By doing this, we can generate images like this.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/color_circle.png" alt="Color circle using barycentric triangle interpolation" class="img-33">
        <figcaption>Color circle using barycentric triangle interpolation</figcaption>
    </figure>
</div>

## Task 5: Texture Mapping via Pixel Sampling
Pixel sampling is how we fetch the correct pixel in the texture to apply it to our render. Given a triangle in pixel space and corresponding triangle in texture space, we can use interpolation to properly map textures onto the triangle, despite the triangles being in different coordinate spaces. For nearest neighbor pixel sampling, we interpolate uv using barycentric coordinates to map the point on the triangle to a point in the texture and round to get the nearest appropriate texel. For bilinear interpolation, we get our pixel-texel mapping and interpolate with the four surrounding texels along two directions to get a weighted average of the colors of the bounding box based on their proximity to the mapped point.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/nearestnosup.png" alt="Nearest neighbor texture mapping no supersampling" class="img-33">
        <figcaption>Nearest-neighbor interpolation, sample rate 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/nearestsuper.png" alt="Nearest neighbor texture mapping supersampling16" class="img-33">
        <figcaption>Nearest-neighbor interpolation, sample rate 16</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/bilinearnosup.png" alt="Bilinear texture mapping no supersampling" class="img-33">
        <figcaption>Bilinear interpolation, sample rate 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/bilinearsuper.png" alt="Bilinear texture mapping supersampling 16" class="img-33">
        <figcaption>Bilinear interpolation, sample rate 16</figcaption>
    </figure>
</div>

The method of pixel sampling will make a larger difference depending on the amount of aliasing occurring in the image. Pixel sampling by bilinear interpolation will tend to smooth out areas of high contrast where aliasing is likely to occur, as with the gridlines on the map. We see that there is a noticeable difference in the images where supersampling is not being done compared to when supersampling is applied.

## Task 6: Texture Mapping via Mipmaps
Level sampling is a technique that uses different resolutions of the same texture to account for distance when texturing an object. For the human eye, the smallest resolvable point in vision is fixed regardless of distance from the viewer. This means that the more distant an object is from the viewer, the area that contributes photons to that point in vision increases such that it is harder to distinguish things from a distance. To achieve this same effect in a simulated environment, we have to aggregate texels in a texture when an object is further from the camera in world space which we can do by downsampling the texture prior to texturing. We can then use these multiple levels of this mipmap which reflect the distance from the camera when texturing an object.

We are given three different methods for level sampling. The L_ZERO method is implemented for us where we texture with the zeroth texture level for all pixels which is the highest resolution texture and the “base” texture that we would expect some aliasing from where distant pixels seem too sharp. The L_NEAREST method uses the following equation and rounds the result to get the appropriate mipmap level.

<div class="math-size-150">
    $$ level = max(\sqrt{dx_u^2 + dx_v^2}, \sqrt{dy_u^2 + dy_v^2}) $$
</div>

The L_LINEAR interpolates between the two nearest mipmap levels and takes the weighted average of the results of getting the texels from both levels.

Supersampling appears to be the method that reduces aliasing the most without otherwise affecting image quality. However, this method does not seem to address the Moiré pattern present in some images such as the one below. This method is also computationally expensive and uses a lot of additional memory.

Bilinear pixel interpolation did not reduce aliasing to the extent that supersampling does even compared to 4 pixel supersampling, but it is effective in reducing aliasing slightly particularly for thin, curved lines as seen below but also aids in smoothing edges with large contrast. This method is fairly expensive since you need to sample four times in the texture and perform three linear interpolation operations to get the color value for a single pixel in screen space compared to nearest neighbor interpolation where two rounding operations are used to find the correct texel. This method does not use much additional memory however.

Level sampling does effectively reduce aliasing as seen below where the lines of the staff are now visually continuous and this method reduces the Moiré pattern the most out of all the antialiasing methods. However this seems to be largely achieved by downsampling the texture as the image looks significantly blurrier compared with the other antialiasing methods since the renderer is likely picking higher levels of the mipmap. Nearest level sampling is relatively cheap computationally compared to the other antialiasing methods since it is mostly arithmetic. However, linear level sampling is more expensive since you need to sample at two different layers and interpolate the result between the two. 

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/lzpn.png" alt="L_ZERO P_NEAREST" class="img-33">
        <figcaption>L_ZERO P_NEAREST</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/lzpl.png" alt="L_ZERO P_LINEAR" class="img-33">
        <figcaption>L_ZERO P_LINEAR</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/lnpn.png" alt="L_NEAREST P_NEAREST" class="img-33">
        <figcaption>L_NEAREST P_NEAREST</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw1/lnpl.png" alt="L_NEAREST P_LINEAR" class="img-33">
        <figcaption>L_NEAREST P_LINEAR</figcaption>
    </figure>
</div>
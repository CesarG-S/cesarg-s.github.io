---
layout: default
permalink: /cs184/hw3/
title: CS184 - Homework 3
---
Cesar Garcia Santana, Ethan Ye
## Overview
In this project we created a raytracer by building up rays, figuring out how they intersected with objects, and calculating the light that bounces off of a surface from a light source or another object. The biggest piece of the puzzle that simplified a lot of the work was getting intersections and rays right, as they automatically figure out and set max_t to the nearest intersection point. This meant we didn't have to perform checks at every intersection to see if it was truly the first intersection or not. Implementing a BVH that splits efficiently was also very useful in reducing render times dramatically, making it possible to render more complex shapes and objects within minutes. Getting the light calculations down involved a lot of reading of the slides to get the right formulas, and properly differentiating between object and world coordinate spaces. Once we figured out the recursion, a lot of the pieces simply fell into place, which was nice, because the code isn't very long at all.

We mostly worked independently until we got to global illumination, at which point we worked together to get the right formulas and review each other's code. Things went well and once we finished with the code, we were able to leverage the power of two computers to render images quickly. We learned a lot about the process of stacking bounces of light to generate an image, and seeing each bounce individually was one of my favorite parts.

## Part 1
## Camera Ray Generation
The goal is to take normalized image coordinates (x, y) in the range [0, 1] and produce a ray in world coordinate space. To do this, we first go from image coordinates to camera sensor space, and then from sensor space to world space using a c2w (camera-to-world) transformation.

The image space to sensor space transformation is easy; the camera sensor has a width spanning tan(hFov) in the x direction and tan(vFov) in the y direction, so we set up the sensor space such that the center of image space (0.5, 0.5) corresponds to (0, 0) in sensor space. This means the left edge is at -0.5tan(hFov) and the right edge is at 0.5tan(hFov) in sensor space, and likewise for the bottom and top edge with vFov. To get values in between, we linearly interpolate for values between 0 and 1. Sensor space is in 3D while image space is in 2D, so we also define the sensor to exist on the Z=-1 plane. It is important to note that sensor space is implicitly defined such that the origin is the camera's position.

Once we have our sensor space vector, we transform to world coordinates using the c2w matrix (matrix-vector multiply). Once we have the camera's position in world coordinates and the new sensor-world direction, we generate a ray such that the origin is the camera's position in world space and the direction is the new transformed sensor-world vector (normalized).

With the ability to generate rays given image coordinates, we can do all sorts of things, like sample rays to estimate lighting, implement primitive-ray intersection detection, and more.

## Triangle and Sphere Intersection Detection
Ray intersections follow a general formula: find an implicit formula for a primitive and plug in the formula for a ray. If there exists a solution, then there may exist a valid intersection point. For a triangle, one way to detect ray intersections is by generating the implicit formula for a plane, plugging in the equation for a ray, and using barycentric coordinates to check whether the intersection point lies inside or outside the triangle.

A more clever method that solves both the intersection  and barycentric coordinates at the same time is the Möller–Trumbore algorithm. This is the route we took for calculating the intersection of a ray and a triangle. This method works by creating an implicit formula for a triangle (barycentric coordinates) and plugging in the equation of a line. By rearranging some terms, one can solve using Cramer's Rule, which translates to a few cross products and dot products.

Calculating the ray intersection with a sphere follows the same general formula described earlier. Plugging in the equation for a ray into the sphere implicit formula generates a quadratic equation, which can be solved using the quadratic formula. If solutions exist, then there are either 1 or 2 intersection points. We use the smaller of the two as the first intersection.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/p1-bench.png" alt="Bench p1" class="img-33">
        <figcaption>bench.png</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/p1-bunny.png" alt="Bunny p1" class="img-33">
        <figcaption>bunny.png</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/p1-gems.png" alt="Gems p1" class="img-33">
        <figcaption>gems.png</figcaption>
    </figure>
</div>

## Part 2
## BVH Construction
A bounding volume hierarchy (BVH) lets us combine primitives into groups with a shared bounding box such that it is quick to traverse and can provide a guarantee for when a ray is certain to not intersect with a primitive. A BVH essentially forms a binary tree of primitive groups, and in our case, splits are determined by comparing the centroid coordinate of the axis with the longest bounding box extent.

In simpler terms, we find the minimum bounding box that holds all primitives and find the longest side. This is the axis that we will split along. We sort the primitives by this axis using their centroid and split the first half into the left group and the second half into the right group. We essentially split by median centroid.

We recursively split each subgroup following the same rules, but with their own smaller bounding boxes. We stop splitting once a group contains no more than max_leaf_size primitives. 

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/dragon.png" alt="Dragon" class="img-33">
        <figcaption>dragon.png</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/wall-e.png" alt="Wall-E" class="img-33">
        <figcaption>wall-e.png</figcaption>
    </figure>
</div>

The times to render dragon.png and wall-e.png were 0.9193 seconds and 1.0771 seconds, respectively. Without BVH acceleration, this would take substantially longer, even with the simple normal shading. The benefit of traversing only the necessary bounding boxes is that it becomes possible to render complex scenes in little time. Simpler objects also benefit from BVH acceleration, as bench, gems, and bunny all took less than 0.9 seconds to render, with the average being around .7 seconds.

## Part 3
## Direct Lighting Implementation
Direct lighting describes light that shines on a point directly from a light source (zero-bounce) and light that shines on a point after having bounced off of a surface from a light source (one-bounce). The sum of both is what we are after.

Zero-bounce is easy - if we find out that a camera ray intersects with an object, we add its emission to our estimate of how illuminated the pixel should be. For a light source, this emission will be high. If we rendered an image with only zero-bounce illumination, we would get pitch black everywhere except for where there are light sources visible to the camera.

One-bounce illumination is the more interesting one here. We implemented two approaches - uniform hemisphere sampling and lighting sampling - to determine the degree to which an object intersected by a ray should be illuminated. With one-bounce illumination, now objects that are in the light become visible to the renderer, since light bounces off of the object and goes towards the camera.

Both approaches start off by shooting a ray from the camera into the scene. If the ray intersects an object, the calculation begins.

## Uniform Hemisphere Sampling
Once we have an intersection point, we uniformly randomly sample a direction on the hemisphere at the intersection point where "up" is the normal vector at the object's surface. We take this direction and create a ray that starts at the intersection point and points in the sampled direction. We then look to see if the ray intersects with a light source. If it does, we use the reflection equation to calculate how much light should reflect from the light source, off the object, and towards the camera. We sample a few uniformly random directions on the hemisphere and take the average light from these rays to get our final lighting estimate for that initial camera ray.

## Lighting Sampling
The problem with uniform hemisphere sampling is that most sampled directions may not lead to a light source. This creates a lot of gaps in the case where no light source is found even if there may have been a path to one. A better method is to sample a direction from each light source that points towards a known intersection point. If light successfully hits the object, then we have found a viable path from light source to object to camera. Again starting from a camera ray intersecting with an object, we iterate over each light and sample a direction from the light and the intersection point. From there, we emit a "shadow ray" starting at the intersection point in the direction from the point to the light and test whether no intersection occurs. If there is nothing blocking the light, then we use the reflection equation to calculate how much light should reflect from the light source, off the object, and towards the camera. We also sample a few directions from the light and take the average light from these rays to get our final lighting estiamate for that initial camera ray.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/p3-bunny_1.png" alt="Bunny 1 sample per light ray" class="img-20">
        <figcaption>Bunny 1 sample per light ray</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/p3-bunny_4.png" alt="Bunny 4 samples per light ray" class="img-20">
        <figcaption>Bunny 4 samples per light ray</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/p3-bunny_16.png" alt="Bunny 16 samples per light ray" class="img-20">
        <figcaption>Bunny 16 samples per light ray</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/p3-bunny_64.png" alt="Bunny 64 samples per light ray" class="img-20">
        <figcaption>Bunny 64 samples per light ray</figcaption>
    </figure>
</div>

The difference between uniform hemisphere sampling and lighting sampling is dramatic. Uniform hemisphere sampling works forward, from intersection point to light, so there is the possibility of holes in the final result, whereas lighting sampling works inversely, from light to intersection point, so each pixel is accounted for, unless there really is no light. Lighting sampling is also just more efficient, as each result is useful, whereas uniform hemisphere sampling often wastes computation by sampling directions that don't lead to light sources.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_hemisphere.png" alt="Bunny uniform hemisphere" class="img-20">
        <figcaption>Bunny uniform hemisphere sampling</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_lighting.png" alt="Bunny lighting sampling" class="img-20">
        <figcaption>Bunny lighting sampling</figcaption>
    </figure>
</div>

## Part 4
## Indirect Lighting Implementation

For indirect lighting, we first set the maximum number of ray bounces for a ray in raytrace_pixel. Then, this method calls est_radiance_global_illumination which calls at_least_one_bounce_radiance. This function first estimates the lighting of the ray at the current recursive depth using either hemisphere or importance sampling and returns it immediately if we are at the maximum recursive depth set previously. Otherwise, we have a 70% chance of continuing recursion where we use sample_f to find a direction in which the ray bounces from the object it intersects, we create a new ray as before and generate an intersection which we use to recursively call at_least_one_bounce_radiance. Once a value is returned from this call, we normalize it by multiplying by bsdf and the cosine of w_in and dividing by the pdf from sample_f and one minus our termination coefficient.

If isAccumBounces is true, we return the current layer’s one_bounce_radiance plus the value from the recursive call. If isAccumBounces is false, we only return the recursive call’s value. Because at_least_one_bounce_radiance does not account for zero bounce radiance, we add that in during est_radiance_global_illumination and for the edge case where the desired number of ray bounces is 0, we only handle that in est_radiance_global_illumination as well. Below are images rendered with global illumination.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/spheres_GI.png" alt="Spheres global illumination" class="img-33">
        <figcaption>./pathtracer -t 8 -s 1024 -l 1 -m 5 -r 480 360 -f spheres_GI.png ../../../dae/sky/CBspheres_lambertian.dae</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_GI.png" alt="Bunny global illumination" class="img-33">
        <figcaption>./pathtracer -t 8 -s 1024 -l 1 -m 5 -r 480 360 -f bunny_GI.png ../../../dae/sky/CBbunny.dae</figcaption>
    </figure>
</div>

Below is CBbunny.dae rendered with only direct illumination and only indirect illumination.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_direct.png" alt="Bunny direct illumination" class="img-33">
        <figcaption>Bunny direct illumination</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_indirect.png" alt="Bunny indirect illumination" class="img-33">
        <figcaption>Bunny indirect illumination</figcaption>
    </figure>
</div>

Below is CBbunny.dae rendered with only the mth bounce of light.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_0.png" alt="Bunny 0 bounce" class="img-20">
        <figcaption>Bunny zero bounce</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_1.png" alt="Bunny 1 bounce" class="img-20">
        <figcaption>Bunny 1st bounce</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_2.png" alt="Bunny 2 bounce" class="img-20">
        <figcaption>Bunny 2nd bounce</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_3.png" alt="Bunny 3 bounce" class="img-20">
        <figcaption>Bunny 3rd bounce</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_4.png" alt="Bunny 4 bounce" class="img-20">
        <figcaption>Bunny 4th bounce</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_5.png" alt="Bunny 5 bounce" class="img-20">
        <figcaption>Bunny 5th bounce</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_0a.png" alt="Bunny 0 bounce acc" class="img-20">
        <figcaption>Bunny zero bounce accumulated</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_1a.png" alt="Bunny 1 bounce acc" class="img-20">
        <figcaption>Bunny 1st bounce accumulated</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_2a.png" alt="Bunny 2 bounce acc" class="img-20">
        <figcaption>Bunny 2nd bounce accumulated</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_3a.png" alt="Bunny 3 bounce acc" class="img-20">
        <figcaption>Bunny 3rd bounce accumulated</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_4a.png" alt="Bunny 4 bounce acc" class="img-20">
        <figcaption>Bunny 4th bounce accumulated</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_100a.png" alt="Bunny 5 bounce acc" class="img-20">
        <figcaption>Bunny 5th bounce accumulated</figcaption>
    </figure>
</div>

The 2nd bounce of light is the layer that primarily contributes to the reflected light effect in the image, lighting up the bottom of the bunny as well as the ceiling which contributes to the realistic feel of the lighting. It also slightly adds color to the bunny which we expect from light bouncing off the colored walls on either side of the bunny further adding to this effect. The overall scene is brighter as well which, without indirect lighting, the scene would feel far too dark despite the bright lighting.

The 3rd bounce of light gives a slight glow to the ground around the bunny as well as its neck which accounts for light bouncing off the bottom of the bunny onto the ground which further contributes to the more realistic lighting of the scene, softening the shadow slightly. There is also some light generally around the scene which makes the whole scene brighter as with the 2nd bounce.

(Russian Roulette Images)
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_0a.png" alt="Bunny 0 bounce" class="img-20">
        <figcaption>Bunny max_ray_depth 0</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_1a.png" alt="Bunny 1 bounce" class="img-20">
        <figcaption>Bunny max_ray_depth 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_2a.png" alt="Bunny 2 bounce" class="img-20">
        <figcaption>Bunny max_ray_depth 2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_3a.png" alt="Bunny 3 bounce" class="img-20">
        <figcaption>Bunny max_ray_depth 3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_4a.png" alt="Bunny 4 bounce" class="img-20">
        <figcaption>Bunny max_ray_depth 4</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_100a.png" alt="Bunny 5 bounce" class="img-20">
        <figcaption>Bunny max_ray_depth 100</figcaption>
    </figure>
</div>

(Varied Sample-Per-Pixel Images)
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_1s.png" alt="Bunny sample-per-pixel 1" class="img-20">
        <figcaption>Bunny sample-per-pixel 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_2s.png" alt="Bunny sample-per-pixel 2" class="img-20">
        <figcaption>Bunny sample-per-pixel 2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_4s.png" alt="Bunny sample-per-pixel 4" class="img-20">
        <figcaption>Bunny sample-per-pixel 4</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_8s.png" alt="Bunny sample-per-pixel 8" class="img-20">
        <figcaption>Bunny sample-per-pixel 8</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_16s.png" alt="Bunny sample-per-pixel 16" class="img-20">
        <figcaption>Bunny sample-per-pixel 16</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_64s.png" alt="Bunny sample-per-pixel 64" class="img-20">
        <figcaption>Bunny sample-per-pixel 64</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_1024s.png" alt="Bunny sample-per-pixel 1024" class="img-20">
        <figcaption>Bunny sample-per-pixel 1024</figcaption>
    </figure>
</div>

## Part 5
## Adaptive Sampling

Adaptive sampling is a sampling method that prevents redundant sampling of areas of fast light convergence such as areas directly exposed to a light source. In raytrace_pixel, we keep a running sum of the illumination of a pixel (s1) and this value squared (s2) to calculate the sample mean and standard deviation. When the count of the samples we have taken is a multiple of samplesPerBatch, we check if the following inequality is true:

<div class="math-size-150">
    $$ 1.96 \cdot \sqrt{\frac{\sigma^2}{n}} \leq \text{maxTolerance} \cdot \mu $$
    
    $$ s_1 = \sum_{k=1}^n x_k $$
    $$ s_2 = \sum_{k=1}^n x_k^2 $$
    $$ \mu = \frac{s_1}{n} $$
    $$ \sigma^2 = \frac{1}{n-1} \cdot (s_2 - \frac{s_1^2}{n}) $$
</div>

If this condition is true, we decide that the color value for the current pixel has converged within our tolerance to its final value such that we set num_samples to the current number of samples taken and stop sampling. We then divide our running sum of the pixel RGB values by num_samples as usual and update the buffer accordingly. Below are examples of raytraced images with sampling rate images.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny.png" alt="Bunny adaptive" class="img-33">
        <figcaption>./pathtracer -t 8 -s 2048 -a 64 0.05 -l 1 -m 5 -r 480 360 -f bunny.png ../../../dae/sky/CBbunny.dae</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/bunny_rate.png" alt="Bunny rate" class="img-33">
        <figcaption>bunny_rate.png</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/spheres.png" alt="Spheres adaptive" class="img-33">
        <figcaption>./pathtracer -t 8 -s 2048 -a 64 0.05 -l 1 -m 5 -r 480 360 -f spheres.png ../../../dae/sky/CBspheres_lambertian.dae</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw3/spheres_rate.png" alt="Spheres rate" class="img-33">
        <figcaption>spheres_rate.png</figcaption>
    </figure>
</div>
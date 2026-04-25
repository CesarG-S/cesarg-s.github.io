---
layout: default
permalink: /cs184/hw4/
title: CS184 - Homework 4
---
Cesar Garcia Santana, Ethan Ye
## Overview
We created a cloth simulation, which combines physics principles and a grid of point masses with attached springs. The big idea is that we have many of the same operation running at the same time each frame, so if we want the simulation to run well in real time, we cannot miss out on optimizations to reduce the runtime of each simulation step. Using a spatial map is a good example of this - instead of running through each pair of point masses to detect self-collisions (O(N^2)), we use a spatial map to perform a small, finite set of checks (on average) for each point mass (O(N)). All of this explains why GPUs are so good at rendering images, and why we can use GPUs to render shaders in video games, for example. This simulation also taught us how hard it can be to create realistic simulations, though it is not too hard to make simulations that look good enough.

## Part 1
## Cloth Masses and Springs
Underlying the cloth simulation will be a grid of point masses and a set of springs connecting them. There are three types of spring constraints: structural constraints between a point mass and its close left or top neighbor; shearing constraints between a point mass and its close upper-left or upper-right neighbor; and bending constraints between a point mass and its two-away left or two-away top neighbor.

Once we create the grid and connect the springs to the point masses, we get a cloth wireframe as shown.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p1-cloth.png" alt="Cloth wireframe" class="img-33">
        <figcaption>Cloth wireframe</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p1-onlyshear.png" alt="Wireframe shear only" class="img-33">
        <figcaption>Shear constraints only</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p1-noshear.png" alt="Wireframe no shear" class="img-33">
        <figcaption>No shear constraints</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p1-all.png" alt="Wireframe all constraints" class="img-33">
        <figcaption>All constraints</figcaption>
    </figure>
</div>

## Part 2
## Cloth Simulation
The actual cloth simulation performs the physics calculations to figure out where to move each point mass and by how much. At each timestep, we want to figure out the forces acting on each point mass, and figure out how the velocity and positions change as a result.

The main forces we track are external forces, like gravity, and spring correction forces, from Hooke's Law.

<div class="math-size-150">
    $$ F = ma $$
    $$ F_s = k_s * (||p_a - p_b|| - l) $$
</div>

We use Verlet integration to compute the new point mass positions at each timestep. Given a damping coefficient d (between 0 and 1), the position at the next timestep is calculated as:

<div class="math-size-150">
    $$ x_{t+dt} = x_t + (1-d) * (x_t - x_{t-dt}) + a_t * dt^2 $$
</div>

The acceleration is obtained after summing the forces acting on a point mass, and we store both its previous and current positions. We do not update a point mass if it is pinned so that it stays in place.

We also want to make sure that springs don't get too deformed, so if the distance between two point masses ends up being 1.1x longer than the spring's natural resting length, we pull them back together (also accounting for the possibility that one point mass is pinned).

Now that we have a basic functioning simulation, we can play around with some parameters.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-default.png" alt="Default pinned 2" class="img-33">
        <figcaption>Default settings (ks=5000 N/m, density=15 g/cm^2)</figcaption>
    </figure>
</div>

A low spring constant (ks) made the cloth "looser"; the forces holding the springs close to its neighbors were much weaker, so the cloth drooped at rest. A high spring constant made the cloth more rigid; the cloth drooped much less at rest and held its shape better near the top where the corners were pinned.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-low-ks.png" alt="Pinned 2 low ks" class="img-33">
        <figcaption>ks = 5 N/m</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-high-ks.png" alt="Pinned 2 high ks" class="img-33">
        <figcaption>ks = 50000 N/m</figcaption>
    </figure>
</div>

Giving the cloth a higher density had a similar effect to lowering the spring constant - the pull of gravity had the strongest force, so the cloth drooped. Lowering the density, as you would expect, made it easier for the springs to hold together the cloth.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-low-density.png" alt="Pinned 2 low density" class="img-33">
        <figcaption>density = 1 g/cm^2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-high-density.png" alt="Pinned 2 high density" class="img-33">
        <figcaption>ks = 150 g/cm^2</figcaption>
    </figure>
</div>

Reducing the damping to 0% made the entire cloth endlessly swing, as it preserved the energy of the initial fall from the starting position. A very low damping value made the cloth swing, but it eventually would settle. A high damping value had the cloth very rigidly fold into place - the cloth settled almost row by row and very slowly.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-zero-damping.png" alt="Pinned 2 no damping" class="img-33">
        <figcaption>No damping</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-low-damping.png" alt="Pinned 2 low damping" class="img-33">
        <figcaption>Low damping</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-high-damping.png" alt="Pinned 2 high damping" class="img-33">
        <figcaption>High damping</figcaption>
    </figure>
</div>

Another visualization, for a cloth with all corners pinned.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p2-pinned4.png" alt="Default pinned 4" class="img-33">
        <figcaption>All corners pinned</figcaption>
    </figure>
</div>

## Part 3
## Cloth Object Collisions
For collisions with spheres, we first find the distance between the current position of the point mass and the origin of the sphere. If this distance exceeds the sphere’s radius, we determine that the point has collided with the sphere. If the point’s position is currently inside the sphere, we take the radius length vector in the direction from the origin to the point mass’ position and add it to the origin’s coordinates to find the tangent point of the sphere. We then subtract the point mass’ last_position from this coordinate to find the correction vector and scale this by 1 - friction.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p3-sphere-ks500.png" alt="Sphere ks=500" class="img-33">
        <figcaption>ks = 500 N/m</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p3-sphere-ks5000.png" alt="Sphere ks=5000" class="img-33">
        <figcaption>ks = 5000 N/m</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p3-sphere-ks50000.png" alt="Sphere ks=50000" class="img-33">
        <figcaption>ks = 50000 N/m</figcaption>
    </figure>
</div>

For collisions with planes, we first calculate the d scalar which represents the distance we need to travel along the vector from last_position to position to get to the point of intersection with the plane. If this value is greater than 0 and greater than or equal to the norm of the vector from last_position to position, we determine that the point has collided with the plane. We use the following equation to find d.

<div class="math-size-150">
    $$ d = \frac{(\text{point} - \text{pm.last_position}) \cdot \text{normal}}{\text{(pm.position - pm.last_position).unit()} \cdot \text{normal}} $$
</div>

We can then find the correction vector as follows.

<div class="math-size-150">
    $$ \text{correction_vector} = ((\text{(1 - friction)} \ast d) - \text{SURFACE_OFFSET}) \ast \text{direction.unit()} $$
</div>

We add this to last_position to find the corrected position for the point mass. In simulate, we loop through every point mass and for every point mass, we loop through every collision object and call collide with that point mass.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p3-plane.png" alt="Plane" class="img-33">
        <figcaption>Cloth sitting on plane</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p3-plane2.png" alt="Plane 2" class="img-33">
        <figcaption>Another view</figcaption>
    </figure>
</div>

## Part 4
## Cloth Self-Collisions
To ensure the cloth doesn't clip into itself, we need to add a repulsion force when two point masses get too close to each other. We can use a spatial map to subdivide the 3D space into zones, and only look for collisions between points if they are in the same zone.

We first build a spatial map using a hash_position that first defines values dx, dy and dz as follows.

<div class="math-size-150">
    $$ dx = 3 \ast \text{width} / \text{num_width_points} $$
    
    $$ dy = 3 \ast \text{height} / \text{num_height_points} $$
    
    $$ dz = \max(dx, dy) $$
</div>

The hashing function then takes a point mass’ position, divides each of its coordinates with the relevant 3D box dimension (3 * width / num_width points for the x coordinate for example), floor this value to effectively determine which box this point mass falls into and then create a unique float identifier using the following:

<div class="math-size-150">
    $$ \text{key} = x + y \ast \text{ceil}(\text{width} / dx) + z \ast \text{ceil}(\text{width} / dx) \ast \text{ceil}(\text{height} / dy) $$
</div>

build_spatial_map loops through all point masses and finds the key for that object using hash_position. It then checks if the count of elements in map with this key are 0. If they are, we initialize map[key] to be a new vector<PointMass*> and then we push_back a reference to the current point mass.

We then loop through all point masses again, this time calling self_collide with each point mass which finds the associated vector in map using this point mass’ key and loops through all point masses that share a 3D box with that point mass using our spatial map. We check if the point mass in this bucket is not the current point mass and if it is within 2 * thickness of the current point mass by taking the norm of the position difference and comparing it to 2 * thickness. If both are true, we find the correction vector by subtracting the position difference norm from 2 * thickness and multiply it by the unit vector in the direction pointing from the other point mass to the current point mass and adding it to a running tally vector along with keeping count of how many self-collisions the current point mass is experiencing. If this point mass has collided with another point mass, we divide the running tally correction vector by the count and divide again by simulation_steps before adding that to the current point mass’ position.

Now that self-collisions are checked, we can visualize our results:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-fall1.png" alt="Cloth fall 1" class="img-33">
        <figcaption>Cloth falling</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-fall2.png" alt="Cloth fall 2" class="img-33">
        <figcaption>Cloth lands</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-fall3.png" alt="Cloth fall 3" class="img-33">
        <figcaption>Cloth begins to unfurl</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-rest.png" alt="Cloth rest" class="img-33">
        <figcaption>Cloth at rest</figcaption>
    </figure>
</div>

We can even see what happens if we tweak a few parameters:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-lowdensity-fall.png" alt="Cloth low density fall" class="img-33">
        <figcaption>Cloth falling; density = 1 g/cm^2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-lowdensity-land.png" alt="Cloth low density land" class="img-33">
        <figcaption>Cloth lands; density = 1 g/cm^2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-lowdensity-rest.png" alt="Cloth low density fall" class="img-33">
        <figcaption>Cloth at rest; density = 1 g/cm^2</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-highdensity-fall.png" alt="Cloth high density fall" class="img-33">
        <figcaption>Cloth falling; density = 150 g/cm^2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-highdensity-land.png" alt="Cloth high density land" class="img-33">
        <figcaption>Cloth lands; density = 150 g/cm^2</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-lowks-fall.png" alt="Cloth low ks fall" class="img-33">
        <figcaption>Cloth falling; ks = 50 N/m</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-lowks-land.png" alt="Cloth low ks land" class="img-33">
        <figcaption>Cloth lands; ks = 50 N/m</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-highks-fall.png" alt="Cloth high ks fall" class="img-33">
        <figcaption>Cloth falling; ks = 50000 N/m</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-highks-land.png" alt="Cloth high ks land" class="img-33">
        <figcaption>Cloth lands; ks = 50000 N/m</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/p4-highks-rest.png" alt="Cloth high ks rest" class="img-33">
        <figcaption>Cloth at rest; ks = 50000 N/m</figcaption>
    </figure>
</div>

\* For low ks and high density, the simulation pushed the cloth through the floor, so there are no "rest" images.

Increasing the density has a similar effect to decreasing ks; the cloth crumples into itself and (presumably) lays at rest while still crumpled.

Decreasing the density has a similar effect to increasing the ks; the cloth almost evenly falls neatly, stacked on top of itself, until it eventually folds out nicely. The cloth does not clash with itself very much, unlike with high density/low ks.

## Part 5
## Shaders
A shader is an image post-processing component that applies changes to the world view after placing the objects in the world, such as - in this case - collision objects and the cloth. They are typically used to apply lighting and shadow, as well as additional texturing for the final render of a frame. Vertex shaders act on the vertices of objects, transforming their geometry in accordance with a predefined map to add texture to an object’s look temporarily. Fragment shaders take in small portions of an object (a fragment), and given information about this fragment, they can apply lighting and color. These shaders then work in tandem to produce various effects while the position of each object is fixed in world space.

The Blinn-Phong shading model works to realistically render light especially for shiny objects by including ambient (reflected from environment), diffuse (scattered), and specular (reflected to camera) lighting components, including an inverse squared distance attenuation coefficient. Below are pictures showing the various components of our Blinn-Phong shader.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-amb.png" alt="Blinn-Phong ambient" class="img-33">
        <figcaption>Blinn-Phong ambient</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-diff.png" alt="Blinn-Phong diffuse" class="img-33">
        <figcaption>Blinn-Phong diffuse</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-spec.png" alt="Blinn-Phong specular" class="img-33">
        <figcaption>Blinn-Phong specular</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-all.png" alt="Blinn-Phong all" class="img-33">
        <figcaption>Blinn-Phong all together</figcaption>
    </figure>
</div>

We show an example using a texture of our own:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-smg.png" alt="SMG texture cloth" class="img-20">
        <figcaption>Textured cloth</figcaption>
    </figure>
</div>

Bump mapping is a shading technique that gives the illusion of detail (bumps) on an object. We also show an example of bump mapping applied to the cloth and sphere:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-bump-cloth.png" alt="Bump mapping cloth" class="img-20">
        <figcaption>Bump mapping: cloth</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-bump-sphere.png" alt="Bump mapping sphere" class="img-20">
        <figcaption>Bump mapping: sphere</figcaption>
    </figure>
</div>

Displacement mapping is similar to bump mapping, but creates depth by actually modifying the positions of the mesh's vertices. We now show an example of displacement mapping on the cloth and sphere:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-disp-cloth.png" alt="Displacement mapping cloth" class="img-20">
        <figcaption>Displacement mapping: cloth</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-disp-sphere.png" alt="Displacement mapping sphere" class="img-20">
        <figcaption>Displacement mapping: sphere</figcaption>
    </figure>
</div>

Bump mapping does not change the geometry of the objects while displacement mapping does. While displacement mapping will then have a more accurate geometry which is visible in the silhouette of the object, it does not work well when the resolution of the meshes are low since if there are only so many vertices to displace and the height map is smooth, the result will appear inappropriately jagged while bump mapping does not have this issue. 

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/lowres-bump.png" alt="Low-res bump mapping" class="img-33">
        <figcaption>Low resolution bump mapping</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/lowres-disp.png" alt="Low-res disp mapping" class="img-33">
        <figcaption>Low resolution displacement mapping</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/highres-bump.png" alt="Low-res bump mapping" class="img-33">
        <figcaption>High resolution bump mapping</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/highres-disp.png" alt="Low-res disp mapping" class="img-33">
        <figcaption>High resolution displacement mapping</figcaption>
    </figure>
</div>

We can see for the low resolution sphere that displacement mapping is not very good, since there are not that many vertices to transform; the loss from the height map is very large. Displacement mapping is better in the high resolution sphere, though bump mapping works about as well for each.

We can also create a shader for a mirror material and apply it to the cloth and sphere. The shader works by approximating incoming radiance using an environment map:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-obama-cloth.png" alt="Mirror cloth" class="img-20">
        <figcaption>Environment-mapped reflections: cloth</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-obama-sphere.png" alt="Mirror sphere" class="img-20">
        <figcaption>Environment-mapped reflections: sphere</figcaption>
    </figure>
</div>

## Extra Credit
We also made a custom shader which we call the noise shader. Our custom shader tries to simulate noise by taking the given parameters and putting it through some functions with some magic numbers as follows:

```cpp
out_color = vec4(rand(v_tangent.xy), rand(v_normal.xy), rand(v_position.xy), 1.0);

float rand(vec2 co) { 
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}
```

It is better seen in a live simulation, but as the simulation view changes, the noise shader applies a view-dependent layer of noise to all pixels belonging to the object. Below are examples of this shader:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-noise-cloth.png" alt="Noise cloth" class="img-20">
        <figcaption>Noise shader: cloth</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw4/5-noise-sphere.png" alt="Noise sphere" class="img-20">
        <figcaption>Noise shader: sphere</figcaption>
    </figure>
</div>
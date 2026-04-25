---
layout: default
permalink: /cs184/hw4/
title: CS184 - Homework 3
---
Cesar Garcia Santana, Ethan Ye
## Overview
text

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

A low spring constant (ks) made the cloth "looser"; the forces holding the springs close to its neighbors were much weaker, so the cloth drooped at rest. A high spring constant made the cloth more rigid; the cloth drooped much less at rest and held its shape better near the top where the corners were pinned.

Giving the cloth a higher density had a similar effect to lowering the spring constant - the pull of gravity had the strongest force, so the cloth drooped. Lowering the density, as you would expect, made it easier for the springs to hold together the cloth.

Reducing the damping to 0% made the entire cloth endlessly swing, as it preserved the energy of the initial fall from the starting position. A very low damping value made the cloth swing, but it eventually would settle. A high damping value had the cloth very rigidly fold into place - the cloth settled almost row by row and very slowly.
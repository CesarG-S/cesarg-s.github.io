---
layout: default
permalink: /cs184/hw2/
title: CS184 - Homework 2
---
Cesar Garcia Santana, Ethan Ye
## Overview
In this project, we built up a way to generate smooth surfaces for rendered objects, starting from Bezier curves, generalizing to Bezier surfaces, and ending up on triangle meshes. We implemented common mesh operations that are useful for increasing or decreasing mesh resolution and implementing smooth lighting while traversing the triangle mesh efficiently. An important theme in this project was making only the minimum necessary edits to the data structures to maintain performance.

## Bezier Curves with de Casteljau Subdivision
De Casteljau's algorithm finds a point that lies on a Bezier curve given a few control points and a parameter t to describe where along the Bezier curve the point lies. It recursively draws lines between each input point and interpolates with t along each line to generate n-1 points (one per line drawn) until there is only one point left. The resulting point lies on the Bezier curve defined by the initial points, so performing de Casteljau's algorithm over all values of t gives you the full Bezier curve.

In action, the subdividing and interpolation process looks like the following:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve.png" alt="Custom Bezier curve" class="img-33">
        <figcaption>Custom 6-point Bezier curve</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-inter1.png" alt="Custom Bezier curve intermediate points 1" class="img-20">
        <figcaption>Intermediate step 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-inter2.png" alt="Custom Bezier curve intermediate points 2" class="img-20">
        <figcaption>Intermediate step 2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-inter3.png" alt="Custom Bezier curve intermediate points 3" class="img-20">
        <figcaption>Intermediate step 3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-inter4.png" alt="Custom Bezier curve intermediate points 4" class="img-20">
        <figcaption>Intermediate step 4</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-inter5.png" alt="Custom Bezier curve intermediate points 5" class="img-20">
        <figcaption>Intermediate step 5</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-inter5-show.png" alt="Custom Bezier curve intermediate points 5 with line" class="img-20">
        <figcaption>Line drawn</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-scroll1.png" alt="Custom Bezier curve different t 1" class="img-33">
        <figcaption>Line drawn, different t</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-scroll2.png" alt="Custom Bezier curve different t 3" class="img-33">
        <figcaption>Line drawn, different t</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-scroll3.png" alt="Custom Bezier curve different t 3" class="img-33">
        <figcaption>Line drawn, different t</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-different1.png" alt="Different Bezier curve 1" class="img-20">
        <figcaption>Different curve, different t</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-different2.png" alt="Different Bezier curve 2" class="img-20">
        <figcaption>Different curve, different t</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p1-bezcurve-different3.png" alt="Different Bezier curve 3" class="img-20">
        <figcaption>Different curve, different t</figcaption>
    </figure>
</div>

## De Casteljau and Bezier Surfaces
De Casteljau's algorithm can be adapted to work in 3D with Bezier surfaces. Instead of working with n control points in 2D, we work with n x n control points in 3D, or essentially n rows of n control points. The idea is similar - for each row, we interpolate using parameter u to find the final point that lies on the Bezier curve for the control points in that row. Once we have all n points for all n rows, we perform one more run of de Casteljau's algorithm with these control points and the parameter v. By running through values of u and v, we can draw out the entire Bezier surface defined by the initial n x n control points.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p2-teapot.png" alt="Teapot bez" class="img-33">
        <figcaption>teapot.bez</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p2-teapot-mesh.png" alt="Teapot bez mesh" class="img-33">
        <figcaption>teapot.bez (with visible mesh)</figcaption>
    </figure>
</div>

## Area-Weighted Vertex Normals
Now working directly with triangle meshes, we implement area-weighted normal vectors to support smooth lighting via Phong shading versus flat shading. To find the normal vector for a triangle, we need to find its three vertices, draw two vectors that define the triangle, and take the cross product to generate the third vector perpendicular to both vectors. This perpendicular vector is the normal vector to the triangle.

To get the area-weighted normal vector at each vertex, we take the normal vectors of all faces incident to the vertex, weight each normal by its triangle's area, and normalize the sum of all such vectors.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p3-teapot-flat.png" alt="Teapot dae flat" class="img-33">
        <figcaption>teapot.dae (flat shading)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p3-teapot-flat-mesh.png" alt="Teapot dae flat mesh" class="img-33">
        <figcaption>teapot.dae (flat shading) (visible mesh)</figcaption>
    </figure>
</div>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p3-teapot-normal.png" alt="Teapot dae Phong" class="img-33">
        <figcaption>teapot.dae (Phong shading)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p3-teapot-normal-mesh.png" alt="Teapot dae flat mesh" class="img-33">
        <figcaption>teapot.dae (Phong shading) (visible mesh)</figcaption>
    </figure>
</div>

## Edge Flip
A common remeshing operation we can do on triangle meshes is edge flipping. In the diagram below, that is the act of taking triangles (b,c,a) and (c,b,d) and swapping the edge such that the resulting triangles are (a,d,c) and (d,a,b).

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p4-edgeflip.jpg" alt="Edge flip diagram" class="img-50">
        <figcaption>Edge flip</figcaption>
    </figure>
</div>

To perform this while adhering to the HalfEdge rules, we must redefine a few next() and twin() relationships. It helps to see all the points and edges involved.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p4-preflip.png" alt="Edge flip before" class="img-33">
        <figcaption>Pre-edge flip labeled</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p4-postflip.png" alt="Edge flip after" class="img-33">
        <figcaption>Post-edge flip labeled</figcaption>
    </figure>
</div>

Fortunately, all half edge and twin relationships remain the same, but half edge and next are affected. Only half edges h0, h1, h2, h3, h4, and h0,t are affected, and their next, edge, vertex, and face are updated accordingly. We also want to make sure that edges, faces, and vertices are assigned to a proper half edge iterator before we finish the swap. 

Below is an example of performing edge flip on a few edges:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p4-teapot-preflip.png" alt="Teapot edge flip before" class="img-33">
        <figcaption>teapot.dae</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p4-teapot-postflip.png" alt="Teapot edge flip after" class="img-33">
        <figcaption>teapot.dae with edge flips</figcaption>
    </figure>
</div>

## Edge Split
Another remeshing operation is edge split. This operation, unlike edge flip, creates new edges and faces, so we must initialize them and fit them into the existing mesh.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-presplit.png" alt="Edge split before" class="img-33">
        <figcaption>Pre-edge split labeled</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-postsplit.png" alt="Edge split after" class="img-33">
        <figcaption>Post-edge split labeled</figcaption>
    </figure>
</div>

In total, three halfedge-twin pairs, two faces, two edges, and one vertex are created. The process is very similar to edge flip - we simply need to take care not to assign the wrong next() and twin() values for all new and existing half edges. Just like in edge flip, existing twin relationships are unaffected, so we only need to worry about the new edge-twin pairs and the edges "within" the split interior. Once we create and populate the proper edges and faces, create the new vertex as the linear interpolation of the end vertices ((vA + vC) / 2), and reassign the necessary next() and twin() pairs, the edge split is complete. 

Below is an example of performing edge split on a few edges:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-teapot-presplit.png" alt="Teapot edge split before" class="img-33">
        <figcaption>teapot.dae</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-teapot-postsplit.png" alt="Teapot edge split after" class="img-33">
        <figcaption>teapot.dae with edge splits</figcaption>
    </figure>
</div>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-teapot-splitflip.png" alt="Teapot edge split flip" class="img-33">
        <figcaption>teapot.dae with edge flips and splits</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-teapot-splitflipsplit.png" alt="Teapot edge split flip split" class="img-33">
        <figcaption>teapot.dae with edge flips and splits</figcaption>
    </figure>
</div>

An interesting thing to note is that the order of flips and splits does have an effect on the final topography of the mesh. I believe the order of flips and splits will matter for upscaling or downscaling if we would like to maintain certain attributes of the mesh, like "evenness" of the distribution of edges.

Boundary triangles are a special case that require additional logic to handle. We don't split boundary polygons, and instead use the following diagram to determine which edges and faces to create:

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-boundary-presplit.png" alt="Boundary edge split before" class="img-33">
        <figcaption>Boundary pre-edge split labeled</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p5-boundary-postsplit.png" alt="Boundary edge split after" class="img-33">
        <figcaption>Boundary post-edge split labeled</figcaption>
    </figure>
</div>

In all, we create two new edges, two halfedge+twin pairs, one face, and one vertex. It is important to ensure that the edge that touches the boundary is properly labeled as such.

## Loop Subdivision
We implemented loop subdivision based on the instructions in the spec. We first calculated the newPosition of each vertex with the positions of its neighbors, using the equations given in the homework spec and set the isNew of these vertices to false. Then, we iterate through all the edges in the mesh, similarly setting their newPosition accordingly and isNew to false. We then iterate through and split all edges where the distinction of new and old edges are handled by our splitEdge method. We iterate through all edges again to check if an edge now connects an old and new edge. If this is the case, we call flipEdge on it. Finally, we set the position of each vertex to their newPosition.

Most meshes stay the same but take on a smoother appearance due to the increased number of triangles. For meshes with sharp edges and corners, these aspects tend to be smoothed out considerably and the mesh loses its shape. For the cube, pre-splitting the edges on the faces of the cube tends to help the mesh maintain its shape. This is due to the vertices of the new polygons after splitting become closer to the edges so the effect of averaging neighboring vertices’ locations becomes less drastic around the sharper edges.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split0-subdiv0.png" alt="Cube e0s0" class="img-20">
        <figcaption>Cube pre-edge-splits=0 subdivisions=0</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split0-subdiv1.png" alt="Cube e0s1" class="img-20">
        <figcaption>Cube pre-edge-splits=0 subdivisions=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split0-subdiv2.png" alt="Cube e0s2" class="img-20">
        <figcaption>Cube pre-edge-splits=0 subdivisions=2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split0-subdiv3.png" alt="Cube e0s2" class="img-20">
        <figcaption>Cube pre-edge-splits=0 subdivisions=3</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split1-subdiv0.png" alt="Cube e1s0" class="img-20">
        <figcaption>Cube pre-edge-splits=1 subdivisions=0</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split1-subdiv1.png" alt="Cube e1s1" class="img-20">
        <figcaption>Cube pre-edge-splits=1 subdivisions=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split1-subdiv2.png" alt="Cube e1s2" class="img-20">
        <figcaption>Cube pre-edge-splits=1 subdivisions=2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split1-subdiv3.png" alt="Cube e1s3" class="img-20">
        <figcaption>Cube pre-edge-splits=1 subdivisions=3</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split2-subdiv0.png" alt="Cube e2s0" class="img-20">
        <figcaption>Cube pre-edge-splits=2 subdivisions=0</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split2-subdiv1.png" alt="Cube e2s1" class="img-20">
        <figcaption>Cube pre-edge-splits=2 subdivisions=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split2-subdiv2.png" alt="Cube e2s2" class="img-20">
        <figcaption>Cube pre-edge-splits=2 subdivisions=2</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-split2-subdiv3.png" alt="Cube e2s3" class="img-20">
        <figcaption>Cube pre-edge-splits=2 subdivisions=3</figcaption>
    </figure>
</div>

Due to how loop subdivision is implemented, the vertices in the original mesh that have less incident edges are allowed to be farther from its neighbors. Thus, the distance from the center of the cube to each corner relative to the average of these distances scales inversely with the number of incident edges to that corner in the original mesh. Because there are six faces each with one edge through it and eight corners in this cube mesh, there is no way to fully alleviate this asymmetry with edge flips alone. The simplest way to guarantee symmetry would be to split each face edge once such that each face looks like it has an “x” through it.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-square-nopresub-split0.png" alt="Square no presplit div 0" class="img-33">
        <figcaption>Square face no pre-split subdivisions=0</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-square-nopresub-split6.png" alt="Square no presplit div 6" class="img-33">
        <figcaption>Square face no pre-split subdivisions=6</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-square-presub-split0.png" alt="Square presplit div 0" class="img-33">
        <figcaption>Square face pre-split subdivisions=0</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/cs184/hw2/p6-square-presub-split6.png" alt="Square presplit div 6" class="img-33">
        <figcaption>Square face pre-split subdivisions=6</figcaption>
    </figure>
</div>
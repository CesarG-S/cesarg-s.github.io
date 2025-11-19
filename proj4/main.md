---
layout: default
permalink: /project4/
title: CS180 - Project 4
---
## Introduction
The goal of this project is to create a NeRF for an object of my choosing.

## Calibration
The first step is to estimate the camera's intrinsics. I did this by printing out a set of 6 ArUco tags and taking phots of them on my table. Using an ArUco detector, I was able to extract the pixel coordinates of the corners and assign them to world coordinates of my choosing. It is important that the same marker gets the same world coordinates.

<div class="image-row"> 
    <img src="/assets/images/proj4/calibrate1.jpg" alt="calibrate1" class="img-20">
    <img src="/assets/images/proj4/calibrate13.jpg" alt="calibrate13" class="img-20">
    <img src="/assets/images/proj4/calibrate72.jpg" alt="calibrate72" class="img-20">
    <img src="/assets/images/proj4/calibrate90.jpg" alt="calibrate90" class="img-20">
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/arucodetect.png" alt="ArUco detect" class="img-33">
        <figcaption>ArUco tag detection</figcaption>
    </figure>
</div>

In total, I had 100 images ready for use in calibration from all sorts of angles and distances. I tried using only a subset of my images for calibration, but later steps suffered as a result of not having data for harder angles, like directly from above.

## Pose Estimation
Now that I had calibrated my camera and extracted the camera's intrinsics using cv2.calibrateCamera(), I could detect ArUco markers in other images and calculate the camera's extrinsics in that photo. This will be the foundation for the NeRF that I build later. The idea at this point is that I can take multiple photos of an object of my choosing, and as long as there is an ArUco tag in the photo, I can calculate the camera's extrinsics and create a new view of my object. It's not too easy to explain in words, so why don't I show you what I mean.

<div class="image-row"> 
    <img src="/assets/images/proj4/render1.png" alt="render1" class="img-33">
    <img src="/assets/images/proj4/render2.png" alt="render2" class="img-33">
    <img src="/assets/images/proj4/render3.png" alt="render3" class="img-33">
    <img src="/assets/images/proj4/render4.png" alt="render4" class="img-33">
</div>

I basically placed an object of my choosing on my table, and placed an ArUco tag next to it. I then took a bunch of photos from different angles, keeping the distance relatively the same, and then calculated the camera's extrinsics using the intrinsics that I calculated earlier and the coordinates of the ArUco tag in the image. With Viser, I can visualize this dome of photo views.

The last step before moving on to the NeRF is to undistort my images and calculate a new intrinsics matrix. The idea is after correcting the distortion for each image, there will be a black border if the shape is kept the same, so we crop the image to keep only the good pixels. This changes the position of the origin, so we need to update the intrinsics matrix. Thankfully, extrinsics and intrinsics are not tied to each other, so we can keep the extrinsics that we calculated for each image and just update the intrinsics of the camera for later use.

## Building a Neural Field
Before we jump straight into a 3D scene, it would be good to try using a neural field to generate a 2D image. Given a set of 2D coordinates, our model outputs a 3D color pixel. By generating a color for each pixel of our canvas, we can rebuild an image. The model is fairly primitive - it uses one image for training and essentially does its best to memorize it. 

The model itself is an MLP with 4 layers that takes in a 2D point, extends it via positional encoding, and sends it through each hidden layer and activation, until it reaches the sigmoid at the end and returns a color pixel.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/MLP.jpg" alt="MLP" class="img-50">
        <figcaption>Four-layer MLP model</figcaption>
    </figure>
</div>

For positional encoding, a 2D point is used as input and a size 2 + 4*L row is generated, inserting high frequency signals where previously only two inputs existed. This helps with the expressiveness of our model, as high frequency inputs make it easier to capture high frequency details. 

My model width is 256 and L equals 10. The model is trained with an Adam optimzer using learning rate 0.01, and uses MSELoss for the loss function.

The following image was used as input to the model.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox.jpg" alt="Fox" class="img-33">
        <figcaption>Input image</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch10.jpg" alt="Fox epoch 10" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch100.jpg" alt="Fox epoch 100" class="img-20">
        <figcaption>Epoch 100</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch1000.jpg" alt="Fox epoch 1000" class="img-20">
        <figcaption>Epoch 1000</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch2999.jpg" alt="Fox epoch 2999" class="img-20">
        <figcaption>Epoch 2999</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_psnr.png" alt="Fox PSNR" class="img-33">
        <figcaption>PSNR vs. Epoch</figcaption>
    </figure>
</div>

With these parameters, the model quickly learns the fox's features and by epoch 1000, can generate a fairly detailed replica.

By choosing different values of L, we can see how exactly positional encoding affects the image generation process.

For L = 3
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch10_L3.jpg" alt="Fox epoch 10 L3" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch100_L3.jpg" alt="Fox epoch 100 L3" class="img-20">
        <figcaption>Epoch 100</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch1000_L3.jpg" alt="Fox epoch 1000 L3" class="img-20">
        <figcaption>Epoch 1000</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch2999_L3.jpg" alt="Fox epoch 2999 L3" class="img-20">
        <figcaption>Epoch 2999</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_psnr_L3.png" alt="Fox PSNR L3" class="img-33">
        <figcaption>PSNR vs. Epoch</figcaption>
    </figure>
</div>

For L=32
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch10_L32.jpg" alt="Fox epoch 10 L32" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch100_L32.jpg" alt="Fox epoch 100 L32" class="img-20">
        <figcaption>Epoch 100</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch1000_L32.jpg" alt="Fox epoch 1000 L32" class="img-20">
        <figcaption>Epoch 1000</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch2999_L32.jpg" alt="Fox epoch 2999 L32" class="img-20">
        <figcaption>Epoch 2999</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_psnr_L32.png" alt="Fox PSNR L32" class="img-33">
        <figcaption>PSNR vs. Epoch</figcaption>
    </figure>
</div>

We can similarly change the model width to see how that affects image generation.

For width = 16
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch10.jpg" alt="Fox epoch 10" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch100.jpg" alt="Fox epoch 100" class="img-20">
        <figcaption>Epoch 100</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch1000.jpg" alt="Fox epoch 1000" class="img-20">
        <figcaption>Epoch 1000</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch2999.jpg" alt="Fox epoch 2999" class="img-20">
        <figcaption>Epoch 2999</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_psnr.png" alt="Fox PSNR" class="img-33">
        <figcaption>PSNR vs. Epoch</figcaption>
    </figure>
</div>

For width = 64
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch10.jpg" alt="Fox epoch 10" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch100.jpg" alt="Fox epoch 100" class="img-20">
        <figcaption>Epoch 100</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch1000.jpg" alt="Fox epoch 1000" class="img-20">
        <figcaption>Epoch 1000</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_epoch2999.jpg" alt="Fox epoch 2999" class="img-20">
        <figcaption>Epoch 2999</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/fox_psnr.png" alt="Fox PSNR" class="img-33">
        <figcaption>PSNR vs. Epoch</figcaption>
    </figure>
</div>

The difference between L=3 and L=10 is rather stark. The difference is most obvious at lower epochs, and even at the final epoch, the L=3 model was substantially blurrier, and only really captured the lower frequencies.

The difference between L=10 and L=32 is much more subtle. The difference is more clear in the epoch 10 image (L=32 represents higher frequencies), but looking at the results for both, they look fairly similar. Turning to the PSNR graphs, it looks like L=10 barely beats L=32, but not by much.

Interestingly, varying the model weights didn't do much to change the PSNR. Additionally, the images look fairly similar. If anything, reducing the weights would serve to speed up computation. It appears that the model does not necessarily need the extra complexity, and the number of epochs all but ensures both models produce similar results.

I also trained the model using an image of my choice.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/palace.jpg" alt="Palace" class="img-33">
        <figcaption>Input image</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/palace_epoch10.jpg" alt="Palace epoch 10" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/palace_epoch100.jpg" alt="Palace epoch 100" class="img-20">
        <figcaption>Epoch 100</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/palace_epoch1000.jpg" alt="Palace epoch 1000" class="img-20">
        <figcaption>Epoch 1000</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/palace_epoch2999.jpg" alt="Palace epoch 2999" class="img-20">
        <figcaption>Epoch 2999</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/palace_psnr.png" alt="Palace PSNR" class="img-33">
        <figcaption>PSNR vs. Epoch</figcaption>
    </figure>
</div>

# Building a Neural Radiance Field
Moving to 3D is much more involved. There are three different coordinate spaces to track, multiple images in the same network, and various distances and angles to account for. Thankfully, this becomes a little manageable once we figure out how to jump between coordinates.

For example, to go from camera coordinates to world coordinates, we can use a rotation matrix and a translation matrix.

More formally, we have:
<div class="math-size-150">
    $$ X_c = T_{w2c}X_w$$

    $$ \begin{bmatrix}  x_c \\
                        y_c \\
                        z_c \\
                        1
    \end{bmatrix} = 
    \begin{bmatrix} R_{3 \times 3} & t \\
                    0_{1 \times 3} & 1
    \end{bmatrix} 
    \begin{bmatrix}     x_w \\
                        y_w \\
                        z_w \\
                        1
    \end{bmatrix} $$
</div>

However, this is the opposite of what we want. Notice that due to the homogeneous coordinates, this expression is equivalent to:
<div class="math-size-150">
    $$ X_c = RX_w + t$$
</div>

Subtracting and taking the inverse of R, we get:
<div class="math-size-150">
    $$ X_w = R^{-1}(X_c - t) $$
    $$ X_w = R^TX_c - R^Tt $$
</div>

Notice that this is basically in the same form as after we removed the homogeneous coordinate, so if we put it back, we get:
<div class="math-size-150">
    $$ X_w = R^TX_c - R^Tt $$

    $$ \begin{bmatrix}  x_w \\
                        y_w \\
                        z_w \\
                        1
    \end{bmatrix} = 
    \begin{bmatrix} R^T_{3 \times 3} & -R^Tt \\
                    0_{1 \times 3} & 1
    \end{bmatrix} 
    \begin{bmatrix}     x_c \\
                        y_c \\
                        z_c \\
                        1
    \end{bmatrix} $$
</div>

And it is this matrix that we define as:
<div class="math-size-150">
    $$ X_w = T_{c2w}X_c $$
</div>

<br>
To go from image coordinates to camera coordinates, we use the camera intrinsics.

More formally, we have:
<div class="math-size-150">
    $$ s \begin{bmatrix}    u \\
                            v \\
                            1 
    \end{bmatrix} = K
    \begin{bmatrix}     x_c \\
                        y_c \\
                        z_c
    \end{bmatrix} $$
</div>

Again, this is the opposite of what we want. We can take the inverse again as follows:
<div class="math-size-150">
    $$ \begin{bmatrix}    x_c \\
                            y_c \\
                            z_c 
    \end{bmatrix} = K^{-1}
    \begin{bmatrix}     su \\
                        sv \\
                        s
    \end{bmatrix} $$
</div>

Now that we can get the world coordinates from camera coordinates, and camera coordinates from image coordinates, we can get the ray corresponding to each pixel in an image. The formula is as follows:
<div class="math-size-150">
    $$ r_d = \frac{X_w - r_o}{\lVert X_w - r_o \rVert_2}$$
</div>

Notice that r_o is simply the translation component of the image extrinsics. To get the X_w associated with a point (u,v), we can get the camera coordinates using s=1, and from there, getting X_w is easy. We just take the norm after subtracting X_w and r_o.

It would also be useful to sample points along the ray. That looks like:
<div class="math-size-150">
    $$ x = r_o + r_d * t $$
</div>
where t is a measure of how far along the ray you are. This does not generate any colors or anything, merely more world coordinates. However, they are the world coordinates that lie on the ray that goes through the original point (u,v), which will be useful later once we train our neural network to sample colors and densities from points.

The neural network in question that we build is the following MLP:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/mlp_nerf.png" alt="NeRF MLP" class="img-50">
        <figcaption>NeRF MLP</figcaption>
    </figure>
</div>
Crucially, I replaced the ReLU at the end of the density branch with softplus, since my values were zeroed out with ReLU.

The last step is to use the model to predict density and color for the samples along the rays we collect. Using the volume rendering equation, we can generate the final color using:
<div class="math-size-150">
    $$ \hat{C}(r) = \sum_{i=1}^N T_i(1-\text{exp}(-\sigma_i\delta_i))c_i $$
    $$ T_i = \text{exp}(-\sum_{j=1}^{i-1}\sigma_j\delta_j)$$
</div>
It is good to know what these values represent. T_i represents the amount of light that is allowed to pass through after having passed through previous points along the ray. It's essentially a measure of how much light is left after it has travelled up the ray.

The volume rendering function volrend takes in as inputs the sigmas (densities), rgbs (colors), and a step size (deltas). The sigmas and rgbs are outputs of the NeRF MLP that we built. The inputs to the MLP are positional encodings of world coordinates that we sampled from rays, which we collected by converting image points to camera coordinates. Everything that we have built up is now important.

## Putting It All Together
Now that we have all the parts, we should put them together and generate some novel image views.

I started with the Lego dataset. It was very nice to work with and quickly produced good results with my MLP. Before I show my results, it helps to have a visual understanding of what the sampling process looks like.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/lego_sampling.png" alt="Lego viser view" class="img-66">
        <figcaption>Lego ray sampling</figcaption>
    </figure>
</div>
The lines coming out of the images are the rays I sample, and the black points along the rays are the points I sampled per ray. It is important to have an idea of how far away the object is from the camera, because if you start sampling too early, you will sample a lot of empty air, but if you sample too late, then you will miss the object entirely.

The final pipeline is as follows:
Sample a number of rays origins and directions from all possible rays from all images. Get points along each ray, and perform positional encoding on the points and directions. Input these into a model, and get out colors and densities. These rgbs and densities are sent to a volumetric rendering function, along with a step size, and an image is generated. This image is compared to a ground truth training image and the MSE loss is calculated and stored. The optimizer resets the gradients, the loss performs backpropagation, and the optimizer tweaks the model parameters, and the cycle repeats until a number of epochs have elapsed.

I collected my results using 96 samples, 2,000 epochs, and a batch size of 5,000.

The results speak for themselves:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/lego_epoch200_val.png" alt="Lego epoch 200" class="img-20">
        <figcaption>Epoch 200</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/lego_epoch400_val.png" alt="Lego epoch 400" class="img-20">
        <figcaption>Epoch 400</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/lego_epoch1000_val.png" alt="Lego epoch 1000" class="img-20">
        <figcaption>Epoch 1000</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/lego_epoch1800_val.png" alt="Lego epoch 1800" class="img-20">
        <figcaption>Epoch 1800</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/lego_graphs.png" alt="Lego graphs" class="img-50">
        <figcaption>Lego PSNR, Validation PSNR, and Loss</figcaption>
    </figure>
</div>

The final thing to do is create artificial cameras, and use the model to generate "in-between" images.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/nerf_lego.gif" alt="Lego spherical rendering" class="img-50">
        <figcaption>Lego spherical rendering</figcaption>
    </figure>
</div>

## Attempt on Own Object
For several days, I tried, but ultimately failed to produce a good NeRF result. I tweaked some of my parameters to fit my GPU's memory, and was able to train a lot of my models in 10-15 minutes. Each epoch took roughly .40 seconds, and I ran 3000 epochs for my own object. Other hyperparameters remained the same. Nevertheless, the image generation wasn't very successful. 

The effect is still pretty interesting. Though what else is interesting is the fact that the validation PSNR improved and then got much worse over time. This is a sign of overfitting, and the model not really learning, which is unfortunate.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch200_val.png" alt="Lego epoch 200" class="img-20">
        <figcaption>Epoch 200</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch400_val.png" alt="Lego epoch 400" class="img-20">
        <figcaption>Epoch 400</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch600_val.png" alt="Ralsei epoch 600" class="img-20">
        <figcaption>Epoch 600</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch800_val.png" alt="Ralsei epoch 800" class="img-20">
        <figcaption>Epoch 800</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch1200_val.png" alt="Ralsei epoch 1200" class="img-20">
        <figcaption>Epoch 1200</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch1600_val.png" alt="Ralsei epoch 1600" class="img-20">
        <figcaption>Epoch 1600</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch2200_val.png" alt="Ralsei epoch 2200" class="img-20">
        <figcaption>Epoch 2200</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_epoch3000_val.png" alt="Ralsei epoch 3000" class="img-20">
        <figcaption>Epoch 3000</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/object_graphs.png" alt="Lego graphs" class="img-50">
        <figcaption>Object PSNR, Validation PSNR, and Loss</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj4/nerf_object.gif" alt="Ralsei spherical rendering" class="img-50">
        <figcaption>Plushie spherical rendering</figcaption>
    </figure>
</div>

It's funny, you can see the ArUco tag spinning in there. But alas, I did not produce a satisfying image. I may revisit this at a future date.
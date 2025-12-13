---
layout: default
permalink: /project5/
title: CS180 - Project 5
---
## Introduction
The goal of this project is to deploy a diffusion model for image generation.

## Noising and Denoising
In diffusion, the idea is to train a model to generate a clean image given a noisy image. It essentially learns how to undo the noise in an image and push an image to the image manifold.

The model I will use is a diffusion pipeline provided by DeepFloyd. The first stage transforms the noisy inputs into an actual image, while the second stage upscales the image from 64x64 to 256x256.

Here are some examples of images it can generate. You first have to convert your prompts into prompt embeddings, but after that, it's very simple. I used seed=1987 before generating my images. The number of inference steps your model takes affects how good the images look. These images were created with num_inference_steps=20.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_floyd.png" alt="Campanile steps=20" class="img-20">
        <figcaption>A picture of the Berkeley Campanile</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campfire_floyd.png" alt="Campanile steps=20" class="img-20">
        <figcaption>An oil painting of people around a campfire</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheesesteak_floyd.png" alt="Campanile steps=20" class="img-20">
        <figcaption>A man eating a 5 foot long cheesesteak</figcaption>
    </figure>
</div>
Not bad, but these images were created with num_inference_steps=200.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_floyd200.png" alt="Campanile steps=200" class="img-20">
        <figcaption>A picture of the Berkeley Campanile</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campfire_floyd200.png" alt="Campanile steps=200" class="img-20">
        <figcaption>An oil painting of people around a campfire</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheesesteak_floyd200.png" alt="Campanile steps=200" class="img-20">
        <figcaption>A man eating a 5 foot long cheesesteak</figcaption>
    </figure>
</div>

Great, now we have an idea of what we're working with. Before we can denoise an image, it helps to be able to add noise to an image.

The forward process is as follows:
<div class="math-size-150">
    $$ x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon \quad \text{where}~ \epsilon \sim N(0, 1)$$
</div>

Given a clean image x_0, we get the noisy image x_t at timestep t by taking a weighted sum of x_0 and noise. The alpha values are determined at each timestep by the model scheduler. My timesteps are such that t=0 is a clean image, and t=1000 is a pure noise image.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile.png" alt="Campanile clean" class="img-10">
        <figcaption>Campanile (clean)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_250.png" alt="Campanile t=250" class="img-10">
        <figcaption>Campanile (t=250)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_500.png" alt="Campanile t=500" class="img-10">
        <figcaption>Campanile (t=500)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_750.png" alt="Campanile t=750" class="img-10">
        <figcaption>Campanile (t=750)</figcaption>
    </figure>
</div>

We have to figure out how to remove the noise from these noisy images to obtain a clean image. The first thing that comes to mind is blurring the high frequencies with a low pass filter. I used a simple Gaussian blur with kernel size 3.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_250.png" alt="Campanile t=250" class="img-10">
        <figcaption>Campanile (t=250)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_250_blur.png" alt="Campanile t=250 blur" class="img-10">
        <figcaption>Campanile (t=250) blurred</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_500.png" alt="Campanile t=500" class="img-10">
        <figcaption>Campanile (t=500)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_500_blur.png" alt="Campanile t=500 blur" class="img-10">
        <figcaption>Campanile (t=500) blurred</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_750.png" alt="Campanile t=750" class="img-10">
        <figcaption>Campanile (t=750)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_750_blur.png" alt="Campanile t=750 blur" class="img-10">
        <figcaption>Campanile (t=750) blurred</figcaption>
    </figure>
</div>

Okay, that's not great. The t=750 image visually blends in, but I suppose you can kind of see it if you squint really hard (but it only works if you know what you're looking for). Instead, I will leverage the already-trained diffusion model to predict the noise given the noisy image. 

The model gives me the predicted noise, so to remove it from the noisy image, it is tempting to just subtract the two, but if we go back to the original forward step, we see that the noisy image is a weighted sum of the original image and random noise. By rearranging terms, we see that the proper way to subtract the noise is as follows:
<div class="math-size-150">
    $$ x_0 = \frac{x_t - \sqrt{1-\bar{\alpha_t}}\epsilon}{\sqrt{\bar{\alpha_t}}} $$
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile.png" alt="Campanile clean" class="img-10">
        <figcaption>Campanile</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_250.png" alt="Campanile t=250" class="img-10">
        <figcaption>Campanile (t=250)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_250_denoised.png" alt="Campanile t=250 denoised" class="img-10">
        <figcaption>Campanile (t=250) denoised</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile.png" alt="Campanile clean" class="img-10">
        <figcaption>Campanile</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_500.png" alt="Campanile t=500" class="img-10">
        <figcaption>Campanile (t=500)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_500_denoised.png" alt="Campanile t=500 denoised" class="img-10">
        <figcaption>Campanile (t=500) denoised</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile.png" alt="Campanile clean" class="img-10">
        <figcaption>Campanile</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_750.png" alt="Campanile t=750" class="img-10">
        <figcaption>Campanile (t=750)</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_750_denoised.png" alt="Campanile t=750 denoised" class="img-10">
        <figcaption>Campanile (t=750) denoised</figcaption>
    </figure>
</div>

With this approach, the model essentially performs one step of denoising, which for t=250, was not too bad, but that's because the input image wasn't too noisy to begin with. A true noisy image would look closer to the t=750 image, which we saw didn't perfectly denoise. There is a strange artifact near the top of the tower, and the image overall is very blurry. If the goal is to "draw a line" between the noisy image and the target clean image, then jumping from one end to the other in a single hop is likely very hard to learn. However, if we iteratively walk along the line, the diffusion model will have an easier time reaching a clean image, since this is what it is trained to do.

Instead of going from t=1000 to t=0 one step at a time, I will traverse my timesteps with a stride of 30, starting from 990 instead of 1000.

To get from an image x_t to x_t', we iterate as follows:
<div class="math-size-150">
    $$ x_{t'} = \frac{\sqrt{\bar\alpha_{t'}}\beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t'})}{1 - \bar\alpha_t} x_t + v_\sigma $$
</div>

x_t represents the current iteration's image (output of previous iteration, or the input noisy image on the first iteration), and x_0 represents the current iteration's predicted clean image, which is just the output of the previous iteration after denoising. x_t' is the noisy image at timestep t'. 

Other values of interest include:
<div class="math-size-150">
    $$ \alpha_t = \frac{\bar{\alpha_t}}{\bar{\alpha_{t'}}} \quad \beta_t = 1 - \alpha_t $$

    $$ v_\sigma~ \text{is random predicted noise} $$
</div>

We try iterative denoising on our image of the Campanile:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_i10.png" alt="Campanile 450" class="img-10">
        <figcaption>Campanile at t=450</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_i15.png" alt="Campanile 600" class="img-10">
        <figcaption>Campanile at t=600</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_i20.png" alt="Campanile 750" class="img-10">
        <figcaption>Campanile at t=750</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_i25.png" alt="Campanile 900" class="img-10">
        <figcaption>Campanile at t=900</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_i30.png" alt="Campanile 990" class="img-10">
        <figcaption>Campanile denoised iteratively</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_onestep.png" alt="Campanile onestep" class="img-10">
        <figcaption>Campanile denoised in one shot</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_unblurred.png" alt="Campanile unblur" class="img-10">
        <figcaption>Campanile noise unblurred</figcaption>
    </figure>
</div>

Since we now have an iterative denoising loop, it makes sense to try starting from pure noise to see what the model generates.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/creature.png" alt="Creature" class="img-20">
        <figcaption>Creature</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/dog.png" alt="Dog" class="img-20">
        <figcaption>Dog</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/dogpig.png" alt="Dog pig" class="img-20">
        <figcaption>Pig? Dog?</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/man4.png" alt="Man" class="img-20">
        <figcaption>Man</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sunset2.png" alt="Sunset" class="img-20">
        <figcaption>Sunset</figcaption>
    </figure>
</div>

That first image is super scary. Everything else is definitely recognizable as something, but it is clear that there is something wrong with a few of these images. Nevertheless, this is a pretty good start. To improve our image generation, we use Classifier-Free Guidance, which generates a noise estimate given a text prompt as conditioning, a noise estimate without conditioning, and combines the two to get the new noise estimate for our noisy input. That is, given conditioned noise and unconditioned noise, we get our noise estimate as follows:
<div class="math-size-150">
    $$ \epsilon = \epsilon_u + \gamma (\epsilon_c - \epsilon_u) $$
</div>

Gamma is a value that controls the strength of CFG. Interestingly, when gamma > 1, image generation quality gets a lot better.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cfg_couple.png" alt="CFG couple" class="img-20">
        <figcaption>CFG Couple</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cfg_dog.png" alt="CFG dog" class="img-20">
        <figcaption>CFG Dog</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cfg_man.png" alt="CFG man" class="img-20">
        <figcaption>CFG Man</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cfg_woman2.png" alt="CFG woman2" class="img-20">
        <figcaption>CFG Woman</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cfg_woman3.png" alt="CFG woman3" class="img-20">
        <figcaption>CFG Woman</figcaption>
    </figure>
</div>


## SDEdit
So we have shown that we can generate an image from random noise, but what if the input wasn't random noise, but a known noisy object? If we start with, say, a noisy picture of the Campanile, but feed the model with a prompt like "a high quality photo", the hope is that we generate an image of something that looks like the Campanile. More generally, we can translate between two similar images if we start with a noisy version and a text prompt. If the image that we start with is less noisy, then the output will look more like the start, but if the image we start with is very noisy, there is a good chance that we end up with a very different (yet subtly similar) output image. To demonstrate this, I generated a few images starting with an image of the Campanile at varying noise levels (represented by i_start, AKA which strided timestep you start at; larger i_start=less initial noise).

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/img2img_t1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Campanile i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/img2img_t3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Campanile i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/img2img_t5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Campanile i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/img2img_t7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Campanile i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/img2img_t10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Campanile i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/img2img_t20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Campanile i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile.png" alt="Campanile" class="img-20">
        <figcaption>Campanile</figcaption>
    </figure>
</div>

Note that the images that I generated were upscaled to 256x256 using the DeepFloyd second stage, but input images are only 64x64, which is why the image of the Campanile looks so blurry by comparison.

I tried a few images of my own as well.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Playground i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Playground i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Playground i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Playground i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Playground i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Playground i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground.png" alt="Playground" class="img-20">
        <figcaption>Playground</figcaption>
    </figure>
</div>
<br><br>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Tree i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Tree i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Tree i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Tree i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Tree i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Tree i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree.png" alt="Tree" class="img-20">
        <figcaption>Tree</figcaption>
    </figure>
</div>

The jump from i_start=10 to 20 is rather impressive. You can tell that for smaller i_start values, aspects of the input are captured, even if the images are of totally different things. For example, the boy in i_start=5 is wearing red for the playground, and the children in i_start=7 and i_start=10 are wearing red clothes and dark green hats. The same thing happens for the tree image, but moreso as the grass or the sky becoming similar to the input, and then it just turning into a tree at i_start=20.

The fun part of being able to transform one image into another similar image is that you can input a hand-drawn image and generate something flashier or more realistic.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sketchdog_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Dog i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sketchdog_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Dog i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sketchdog_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Dog i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sketchdog_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Dog i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sketchdog_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Dog i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sketchdog_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Dog i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/sketchdog.png" alt="Tree" class="img-20">
        <figcaption>Dog sketch</figcaption>
    </figure>
</div>
<br><br>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/luffy_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Pirate i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/luffy_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Pirate i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/luffy_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Pirate i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/luffy_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Pirate i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/luffy_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Pirate i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/luffy_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Pirate i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/luffy.png" alt="Tree" class="img-20">
        <figcaption>Pirate sketch</figcaption>
    </figure>
</div>
<br><br>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheeseburger_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Cheeseburger i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheeseburger_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Cheeseburger i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheeseburger_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Cheeseburger i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheeseburger_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Cheeseburger i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheeseburger_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Cheeseburger i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheeseburger_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Cheeseburger i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/cheeseburger.png" alt="Tree" class="img-20">
        <figcaption>Cheeseburger sketch</figcaption>
    </figure>
</div>

Something else we can try is masking a patch of our image during image generation to influence what the model creates. For example, if we mask out everything but the tip of the Campanile during generation (telling the model to ignore everything but the tip), then we can generate an image of the Campanile but with a different tip. The trick is to create a noisy version of the part of the image that we want to keep, and add it to random noise in the area where we want the model to fill in the gap. The denoising step will keep everything mostly the same except for where you want to fill in.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile.png" alt="Campanile" class="img-20">
        <figcaption>Campanile</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_mask.png" alt="Campanile mask" class="img-20">
        <figcaption>Campanile mask</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_toreplace.png" alt="Campanile to replace" class="img-20">
        <figcaption>Campanile hole to fill</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile_inpaint.png" alt="Campanile inpaint" class="img-20">
        <figcaption>Campanile inpainted</figcaption>
    </figure>
</div>

I did the same for a few more of my images.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground.png" alt="Playground" class="img-20">
        <figcaption>Playground</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_mask.png" alt="Playground mask" class="img-20">
        <figcaption>Playground mask</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_toreplace.png" alt="Playground to replace" class="img-20">
        <figcaption>Playground hole to fill</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground_inpaint.png" alt="Playground inpaint" class="img-20">
        <figcaption>Playground inpainted</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/webdog.png" alt="Dog" class="img-20">
        <figcaption>Dog</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/webdog_mask.png" alt="Dog mask" class="img-20">
        <figcaption>Dog mask</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/webdog_toreplace.png" alt="Dog to replace" class="img-20">
        <figcaption>Dog hole to fill</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/webdog_inpaint.png" alt="Dog inpaint" class="img-20">
        <figcaption>Dog inpainted...?</figcaption>
    </figure>
</div>

Woah, something really strange happened with that dog image. The playground was actually fairly good, but something went really wrong with the dog image. I decided to keep it in because it was really funny. It's hard to say what went wrong, but possibly there was too much new content to generate, and the surrounding pixels weren't very helpful since they are all rather bright.

Now it only makes sense to try using a prompt other than "a high quality photo" in tandem with the image of the Campanile. Using the prompt "a pencil" produces rather cool results.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/pencil_campanile_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Campanile; prompt:"a pencil" i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/pencil_campanile_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Campanile; prompt:"a pencil" i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/pencil_campanile_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Campanile; prompt:"a pencil" i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/pencil_campanile_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Campanile; prompt:"a pencil" i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/pencil_campanile_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Campanile; prompt:"a pencil" i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/pencil_campanile_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Campanile; prompt:"a pencil" i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/campanile.png" alt="Campanile" class="img-20">
        <figcaption>Campanile</figcaption>
    </figure>
</div>

I tried a few other image+prompt combinations, too.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/milk_playground_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Playground; prompt:"a carton of milk" i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/milk_playground_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Playground; prompt:"a carton of milk" i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/milk_playground_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Playground; prompt:"a carton of milk" i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/milk_playground_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Playground; prompt:"a carton of milk" i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/milk_playground_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Playground; prompt:"a carton of milk" i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/milk_playground_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Playground; prompt:"a carton of milk" i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/playground.png" alt="Playground" class="img-20">
        <figcaption>Playground</figcaption>
    </figure>
</div>
<br><br>
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/strawberry_tree_sde1.png" alt="SDEdit 1" class="img-20">
        <figcaption>Tree; prompt:"a strawberry" i_start=1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/strawberry_tree_sde3.png" alt="SDEdit 3" class="img-20">
        <figcaption>Tree; prompt:"a strawberry" i_start=3</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/strawberry_tree_sde5.png" alt="SDEdit 5" class="img-20">
        <figcaption>Tree; prompt:"a strawberry" i_start=5</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/strawberry_tree_sde7.png" alt="SDEdit 7" class="img-20">
        <figcaption>Tree; prompt:"a strawberry" i_start=7</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/strawberry_tree_sde10.png" alt="SDEdit 10" class="img-20">
        <figcaption>Tree; prompt:"a strawberry" i_start=10</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/strawberry_tree_sde20.png" alt="SDEdit 20" class="img-20">
        <figcaption>Tree; prompt:"a strawberry" i_start=20</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/tree.png" alt="Tree" class="img-20">
        <figcaption>Tree</figcaption>
    </figure>
</div>

We can get a bit more creative. Since we can modify the noise as we please, what if we took the noise estimates from two prompts and combined them together? If we take the noise estimate from an image with prompt A, and the noise estimate of the same image, but flipped vertically, with prompt B, and then flipped the flipped noise estimate, and averaged them, we can create a visual anagram, which is an image that appears to look like something else when viewed upside-down, for example. 

For this one, you'll need to hover over the image to see what it looks like flipped vertically.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj5/campfire_oldman_flipped.png" alt="Old man campfire" class="img-33 hover-image">
            <img src="/assets/images/proj5/campfire_oldman.png" alt="Campfire old man" class="img-33 default-image">
        </div>
        <figcaption>Campfire that upside-down is an old man</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <div class="image-hover-swap">
            <img src="/assets/images/proj5/skull_waterfall4_flipped.png" alt="Skull waterfall" class="img-33 hover-image">
            <img src="/assets/images/proj5/skull_waterfall4.png" alt="Waterfall skull" class="img-33 default-image">
        </div>
        <figcaption>Waterfall that upside-down is a skull</figcaption>
    </figure>
</div>

I absolutely love the campfire photo. It is such a good illusion. The skull is a little too obvious, but if you get close, you definitely see the waterfall.

Another illusion you can do is similar to the hybrid images of project 2. The idea is you sum the noise estimate of the image with prompt A under a lowpass filter with the noise estimate of the image with prompt B under a highpass filter. After you remove the noise, the illusion should become apparent when you stand far away.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/skull_waterfall.png" alt="Skull waterfall hybrid" class="img-20">
        <figcaption>Skull + waterfall hybrid</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/skull_waterfall2.png" alt="Skull waterfall hybrid 2" class="img-20">
        <figcaption>Skull + waterfall hybrid</figcaption>
    </figure>
</div>
Two very different images but with the same illusion. Very entertaining.

## Building the actual UNet
So we went in and manipulated a bunch of images, but we used a model that was given to us. Now it is my turn to actually build the UNet that does the image generation.

This is the UNet that I will build.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/unet_chart.png" alt="Unet" class="img-50">
        <figcaption>UNet</figcaption>
    </figure>
</div>

We can then train our denoiser rather easily. We use the MNIST handwritten digit dataset and a simple L2 loss to train the model. For each step of the training loop, we generate a noisy image z from a clean image x as follows:
<div class="math-size-150">
    $$ z = x + \sigma\epsilon \quad \text{where}~ \epsilon \sim N(0, I) $$
</div>

I take an image from the MNIST dataset and show the result of adding noise for varying noise (sigma) levels:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/noise_process_output.png" alt="Noising" class="img-50">
        <figcaption>Noisy image output for varying sigma values</figcaption>
    </figure>
</div>

Now we can train our model using sigma = 0.5, the Adam optimizer with lr = 1e-4, and hidden dimension D = 128 in our UNet. The results are as follows after 1 epoch and after 5 epochs:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/simple_training_output1.png" alt="Training 1 epoch" class="img-20">
        <figcaption>Input and outputs after 1 epoch</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/simple_training_output5.png" alt="Noising" class="img-20">
        <figcaption>Inputs and outputs after 5 epochs</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/simple_training_loss_curve.png" alt="Training loss" class="img-33">
        <figcaption>MSELoss per batch for 5 epochs</figcaption>
    </figure>
</div>

Since we trained our model based on a noise value (sigma) of 0.5, it would be interesting to see how it performs for varying levels of noise. The results are not too surprising:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/out_of_distribution_eval.png" alt="Out of distribution" class="img-50">
        <figcaption>Out of distribution testing</figcaption>
    </figure>
</div>

The model is already trained to handle a noise level of 0.5, so anything below is encompassed by it. However, the model struggles a little more for noise levels above 0.5, but still does a fairly good job at getting a recognizable digit out, though sigma=0.8 did particularly bad.

If we train our UNet to denoise pure noise, something interesting happens:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/pure_noise_denoise_loss_curve.png" alt="Training loss" class="img-33">
        <figcaption>MSELoss per batch for 5 epochs</figcaption>
    </figure>
</div>

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/denoise_pure_noise1.png" alt="Training 1 epoch" class="img-50">
        <figcaption>Outputs after 1 epoch</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/denoise_pure_noise5.png" alt="Noising" class="img-50">
        <figcaption>Outputs after 5 epochs</figcaption>
    </figure>
</div>

It is hard to tell, but the model cannot really train on denoising random noise, since it could belong to any digit. What ends up happening is the model learns the underlying patterns for handwritten digits. The difference is subtle for epoch 1 vs. epoch 5, but epoch 5 is slightly less noisy.

## Time-Conditioned UNet
The model we currently have is a bit limited, since it essentially just performs one-step denoising. If we could iteratively denoise, that would be a lot better. If we train a model to predict the flow from our noisy image to a clean image, then we could iteratively walk along the flow to generate a clean image.

We can "draw" a line between a noisy image and a clean image as follows:
<div class="math-size-150">
    $$ x_t = (1-t)x_0 + tx_1 $$
</div>

To find how x_t changes with respect to time, we take the derivative with respect to t:
<div class="math-size-150">
    $$ \frac{d}{dt}x_t = x_1 - x_0 $$
</div>

It is this value that we train our model to approximate. We add time conditioning to our UNet (unflatten becomes unflatten * t1, and so on; the basic idea is multiply by FCBlock(t) at the dotted arrows):
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_chart.png" alt="Time UNet" class="img-50">
        <figcaption>Time-conditioned UNet</figcaption>
    </figure>
</div>

Training the model is a little different now.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_train_loop.png" alt="Time UNet Training" class="img-50">
        <figcaption>Time-conditioned UNet training</figcaption>
    </figure>
</div>

We take an image x_1 from the MNIST dataset, randomly sample a timestep t between 0 and 1, sample a random image x_0, and calculate x_t using the above formula. We then calculate the MSELoss on the actual flow and the predicted flow. I trained my model with batch_size = 64, learning rate = 1e-2, hidden_dim = 64, with an ExponentialLR scheduler, and 10 epochs.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_loss_curve.png" alt="Time UNet Loss Curve" class="img-33">
        <figcaption>MSELoss per batch for 10 epochs</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_loss_curve_per_epoch.png" alt="Time UNet Loss Curve" class="img-50">
        <figcaption>MSELoss per epoch</figcaption>
    </figure>
</div>

After training, I sampled 40 digits from the model for epochs 1, 5, and 10. The sampling algorithm is as follows:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_sampling.png" alt="Time UNet Sampling" class="img-50">
        <figcaption>Time UNet sampling algorithm</figcaption>
    </figure>
</div>

Essentially, you start with x_t, which is a random noisy image. You then evenly walk through the timesteps between 0 and 1 for T timesteps, each of step size 1/T. You iteratively add to x_t the predicted flow at timestep t weighted by 1/T.

<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_epoch1.png" alt="Time UNet Sample epoch 1" class="img-20">
        <figcaption>Epoch 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_epoch5.png" alt="Time UNet Sample epoch 5" class="img-20">
        <figcaption>Epoch 5</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/time_unet_epoch10.png" alt="Time UNet Sample epoch 10" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
</div>

## Class-Conditioned UNet
There are discernable digits, but there are also a few indiscernable scribbles. To improve the samples, we can condition by class in addition to time. The model itself does not change too drastically - it simply also takes in a digit representing the class (one-hot vector for a digit 0-9) for conditioning. To make the model a little more sophisticated, we implement dropout, where 10% of the time, we set the class conditioning vector to 0 (not the class 0). Conditioning c is similar to conditioning t, but instead of unflatten * t1, we do c1 * unflatten + t1, and so on, where c is the output of FCBlock(c).

The training loop is almost identical to the time-conditioned UNet loop:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/class_unet_train_loop.png" alt="Class UNet Training" class="img-50">
        <figcaption>Class-conditioned UNet training</figcaption>
    </figure>
</div>

The major differences are you make c into a one-hot vector, and zero it out with probability p_uncond. The model also takes in c as an input, but everything else is the same. The hyperparameters were the same as in the time-conditioned UNet.
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/class_unet_loss_curve.png" alt="Class UNet Loss Curve" class="img-33">
        <figcaption>MSELoss per batch for 10 epochs</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/class_unet_loss_curve_per_epoch.png" alt="Class UNet Loss Curve" class="img-50">
        <figcaption>MSELoss per epoch</figcaption>
    </figure>
</div>

The sampling algorithm is also different:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/class_unet_sampling.png" alt="Class UNet Sampling" class="img-50">
        <figcaption>Class UNet sampling algorithm</figcaption>
    </figure>
</div>

The algorithm incorporates classifier-free guidance, as we use the model to generate an conditional flow (input c vector for class) and an unconditional flow (input zero vector for class), take a combination of the flows weighted by gamma (I used gamma = 5), and iteratively add to an initial random noise image for timesteps between 0 and 1 of step size 1/T.

I made my model output four instances of each digit, and here they are:
<div class="image-row"> 
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/class_unet_epoch1.png" alt="Class UNet Sample epoch 1" class="img-20">
        <figcaption>Epoch 1</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/class_unet_epoch5.png" alt="Class UNet Sample epoch 5" class="img-20">
        <figcaption>Epoch 5</figcaption>
    </figure>
    <figure class="image-with-subtitle">
        <img src="/assets/images/proj5/class_unet_epoch10.png" alt="Class UNet Sample epoch 10" class="img-20">
        <figcaption>Epoch 10</figcaption>
    </figure>
</div>

Out of curiosity, I removed the scheduler, and the resulting images were of worse quality. If I had more time, I would definitely try to improve the model without, but to be completely frank, having it made the digit quality a lot better, so you might as well keep it.
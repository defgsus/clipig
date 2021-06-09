# Transforms

Transforms shape the area of the trained image before showing
it to [CLIP](https://github.com/openai/CLIP/) for evaluation. 

All [transforms](reference.md#targetstransforms) that do not change the 
resolution of the image are also available as 
[post processing](reference.md#postproc) effects.
  
Here's a list of all available transformations:

- [add](reference.md#targetstransformsadd): Adds a fixed value to all pixels.
- [blur](reference.md#targetstransformsblur): A gaussian blur is applied to the pixels.
- [border](reference.md#targetstransformsborder): Draws a border on the edge of the image. The resolution is not changed.
- [bwnoise](reference.md#targetstransformsbwnoise): Adds gray-scale noise to the image.
- [center_crop](reference.md#targetstransformscenter_crop): Crops an image of the given resolution from the center.
- [clamp](reference.md#targetstransformsclamp): Clamps the pixels into a fixed range.
- [crop](reference.md#targetstransformscrop): Crops a specified section from the image.
- [edge](reference.md#targetstransformsedge): This removes everything except edges and generally has a bad effect on image
    quality. It might be useful, however.
- [fnoise](reference.md#targetstransformsfnoise): Adds noise to the image's fourier space.
- [mean](reference.md#targetstransformsmean): Adjust the mean color value.
- [mul](reference.md#targetstransformsmul): Multiplies all pixels by a fixed value.
- [noise](reference.md#targetstransformsnoise): Adds noise to the image.
- [pad](reference.md#targetstransformspad): Pads the image with additional pixels at the borders.
- [random_crop](reference.md#targetstransformsrandom_crop): Crops a section of the specified resolution from a random position in the image.
- [random_rotate](reference.md#targetstransformsrandom_rotate): Randomly rotates the image.
- [random_scale](reference.md#targetstransformsrandom_scale): Randomly scales an image in the range specified.
- [random_shift](reference.md#targetstransformsrandom_shift): This randomly translates the pixels of the image.
- [random_translate](reference.md#targetstransformsrandom_translate): Randomly translates an image in the specified range.
- [repeat](reference.md#targetstransformsrepeat): Repeats the image a number of times in the right and bottom direction.
- [resize](reference.md#targetstransformsresize): The resolution of the image is changed.
- [rnoise](reference.md#targetstransformsrnoise): Adds noise with a different resolution to the image.
- [rotate](reference.md#targetstransformsrotate): Rotates the image.
- [shift](reference.md#targetstransformsshift): This translates the image while wrapping the edges around.

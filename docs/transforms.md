# Transforms

Transforms shape the area of the trained image before showing
it to [CLIP](https://github.com/openai/CLIP/) for evaluation. 
  
Here's a list of all available transformations:

- [add](#targetstransformsadd): Adds a fixed value to all pixels.
- [blur](#targetstransformsblur): A gaussian blur is applied to the pixels.
    See [torchvision gaussian_blur](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.gaussian_blur).
- [border](#targetstransformsborder): Draws a border on the edge of the image. The resolution is not changed.
- [center_crop](#targetstransformscenter_crop): Crops an image of the given resolution from the center.
    See [torchvision center_crop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.center_crop).
- [clamp](#targetstransformsclamp): Clamps the pixels into a fixed range.
- [crop](#targetstransformscrop): Crops a specified section from the image.
- [edge](#targetstransformsedge): This removes everything except edges and generally has a bad effect on image
    quality. It might be useful, however...
- [fnoise](#targetstransformsfnoise): Adds noise to the image's fourier space.
- [mul](#targetstransformsmul): Multiplies all pixels by a fixed value.
- [noise](#targetstransformsnoise): Adds noise to the image.
- [pad](#targetstransformspad): Pads the image with additional pixels at the borders.
- [random_crop](#targetstransformsrandom_crop): Crops a section of the specified resolution from a random position in the image.
    See [torchvision random_crop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.random_crop)
- [random_rotate](#targetstransformsrandom_rotate): Randomly rotates the image.
- [random_scale](#targetstransformsrandom_scale): Randomly scales an image in the range specified.
    See [torchvision RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine).
- [random_shift](#targetstransformsrandom_shift): This randomly translates the pixels of the image.
- [random_translate](#targetstransformsrandom_translate): Randomly translates an image in the specified range.
- [repeat](#targetstransformsrepeat): Repeats the image a number of times in the right and bottom direction.
- [resize](#targetstransformsresize): The resolution of the image is changed.
- [rotate](#targetstransformsrotate): Rotates the image.
- [shift](#targetstransformsshift): This translates the pixels of the image.


All [transforms](reference.md#transforms) that do not change the 
resolution of the image are also available as 
[post processing](reference.md#postproc) effects.
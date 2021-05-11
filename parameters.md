#### verbose

`int` default: **2**

Verbosity level
- `0` = off
- `1` = show progress
- `2` = show statistics

#### output

`str` default: **./**

Directory or filename of the output. 
- If a directory, it must end with `/`. 
  In that case, the filename will be the name of the yaml config file. 
- If a filename, it must end with `.png`. Note that a number is attached to the 
  filename or is automatically increased, if the file already exists.

#### snapshot_interval

`int, float` default: **20.0**

Interval after which a snapshot of the currently trained image is saved. 

A float number specifies the interval in seconds. An integer number specifies 
the interval in number-of-epochs.

#### epochs

`int` default: **300**

The number of training steps before stopping the training, not including batch sizes. 

For example, if the number of epochs is `100` and a target has a batch_size of `10`, 
then `1000` training steps will be performed.

#### start_epoch

`int` default: **0**

The number of epochs to skip in the beginning. 

This is used by the GUI application to continue training after config changes.

#### resolution

`list of length 2 of int` default: **[224, 224]**

Resolution of the image to create. A single number for square images or two numbers for width and height.

#### model

`str` default: **ViT-B/32**

The pre-trained CLIP model to use. Options are `RN50`, `RN101`, `RN50x4`, `ViT-B/32`

The models are downloaded from `openaipublic.azureedge.net` and stored in the current user's cache directory

#### device

`str` default: **auto**

The device to run the training on. Can be `cpu`, `cuda`, `cuda:1` etc.

#### learnrate

`float` default: **1.0**

The learning rate of the optimizer. 

Different optimizers have different learning rates that work well. 
However, this value is scaled *by hand* so that `1.0` translates to 
about the same learning rate for each optimizer. 

The learnrate value is available to other expressions as `lr` or `learnrate`.

#### learnrate_scale

`float` default: **1.0**

A scaling parameter for the actual learning rate.

It's for convenience in the case when learnrate_scale is an expression like `1. - t`. 
The actual learnrate can be overridden with fixed values like `2` or `3` in 
different experiments.

The learnrate_scale value is available to other expressions as `lrs` or `learnrate_scale`.

#### optimizer

`str` default: **adam**

The torch optimizer to perform the gradient descent.

### init

Defines the way, the pixels are initialized. Default is random pixels.

#### init.mean

`list of length 3 of float` default: **[0.5, 0.5, 0.5]**

The mean (brightness) of the initial pixel noise. 

Can be a single number for gray or three numbers for RGB.

#### init.std

`list of length 3 of float` default: **[0.1, 0.1, 0.1]**

The standard deviation (randomness) of the initial pixel noise. 

A single number will be copied to the RGB values.

#### init.image

`str` default: **None**

A filename of an image to use as starting point.

The image will be scaled to the desired resolution if necessary.

#### init.image_tensor

`list` default: **None**

A 3-dimensional matrix of pixel values in the range [0, 1]  

The layout is the same as used in 
[torchvision](https://pytorch.org/vision/stable/index.html), 
namely `[C, H, W]`, where `C` is number of colors (3), 
`H` is height and `W` is width.

This is used by the GUI application to continue training after config changes.

### targets

This is a list of *targets* that define the desired image. 

Most important are the [features](#target-features) where
texts or images are defined which get converted into CLIP
features and then drive the image creation process.

It's possible to add additional [constraints](#targets-constraints)
which alter image creation without using CLIP, 
e.g. the image mean, saturation or gaussian blur.

#### targets.active

`bool` default: **True**

A boolean to turn of the target during development. 

This is just a convenience parameter. To turn of a target
during testing without deleting all the parameters, simply 
put `active: false` inside.

#### targets.name

`str` default: **target**

The name of the target. 

Currently this is just displayed in the statistics dump and has no
functionality.

#### targets.start

`int, float` default: **0.0**

Start frame of the target. 

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### targets.end

`int, float` default: **1.0**

#### targets.weight

`float` default: **1.0**

#### targets.batch_size

`int` default: **1**

#### targets.select

`str` default: **all**

### targets.features

#### targets.features.weight

`float` default: **1.0**

#### targets.features.loss

`str` default: **cosine**

#### targets.features.text

`str` default: **None**

#### targets.features.image

`str` default: **None**

### targets.transforms

#### targets.transforms.add

`list of length 3 of float` default: **None**

### targets.transforms.blur

#### targets.transforms.blur.kernel_size

`list of length 2 of int` default: **[3, 3]**

#### targets.transforms.blur.sigma

`list of length 2 of float` default: **None**

### targets.transforms.border

#### targets.transforms.border.size

`list of length 2 of int` default: **[1, 1]**

#### targets.transforms.border.color

`list of length 3 of float` default: **[1.0, 1.0, 1.0]**

#### targets.transforms.center_crop

`list of length 2 of int` default: **None**

#### targets.transforms.clamp

`list of length 2 of float` default: **None**

### targets.transforms.edge

#### targets.transforms.edge.kernel_size

`list of length 2 of int` default: **[3, 3]**

#### targets.transforms.edge.sigma

`list of length 2 of float` default: **None**

#### targets.transforms.edge.amount

`list of length 3 of float` default: **[1.0, 1.0, 1.0]**

#### targets.transforms.mul

`list of length 3 of float` default: **None**

#### targets.transforms.noise

`list of length 3 of float` default: **None**

#### targets.transforms.random_crop

`list of length 2 of int` default: **None**

### targets.transforms.random_rotate

#### targets.transforms.random_rotate.degree

`list of length 2 of float` default: **[-1, 1]**

#### targets.transforms.random_rotate.center

`list of length 2 of float` default: **[0.5, 0.5]**

#### targets.transforms.random_scale

`list of length 2 of float` default: **None**

#### targets.transforms.random_shift

`list of length 2 of float` default: **None**

#### targets.transforms.random_translate

`list of length 2 of float` default: **None**

#### targets.transforms.repeat

`list of length 2 of int` default: **None**

#### targets.transforms.resize

`list of length 2 of int` default: **None**

### targets.transforms.rotate

#### targets.transforms.rotate.degree

`float` default: **None**

#### targets.transforms.rotate.center

`list of length 2 of float` default: **[0.5, 0.5]**

#### targets.transforms.shift

`list of length 2 of float` default: **None**

### targets.constraints

### targets.constraints.blur

#### targets.constraints.blur.weight

`float` default: **1.0**

#### targets.constraints.blur.kernel_size

`list of length 2 of int` default: **[3, 3]**

#### targets.constraints.blur.sigma

`list of length 2 of float` default: **None**

### targets.constraints.edge_max

#### targets.constraints.edge_max.weight

`float` default: **1.0**

#### targets.constraints.edge_max.above

`list of length 3 of float` default: **None**

#### targets.constraints.edge_max.below

`list of length 3 of float` default: **None**

### targets.constraints.edge_mean

#### targets.constraints.edge_mean.weight

`float` default: **1.0**

#### targets.constraints.edge_mean.above

`list of length 3 of float` default: **None**

#### targets.constraints.edge_mean.below

`list of length 3 of float` default: **None**

### targets.constraints.mean

#### targets.constraints.mean.weight

`float` default: **1.0**

#### targets.constraints.mean.above

`list of length 3 of float` default: **None**

#### targets.constraints.mean.below

`list of length 3 of float` default: **None**

### targets.constraints.saturation

#### targets.constraints.saturation.weight

`float` default: **1.0**

#### targets.constraints.saturation.above

`float` default: **None**

#### targets.constraints.saturation.below

`float` default: **None**

### targets.constraints.std

#### targets.constraints.std.weight

`float` default: **1.0**

#### targets.constraints.std.above

`list of length 3 of float` default: **None**

#### targets.constraints.std.below

`list of length 3 of float` default: **None**

### postproc

#### postproc.active

`bool` default: **True**

#### postproc.start

`int, float` default: **0.0**

#### postproc.end

`int, float` default: **1.0**

#### postproc.add

`list of length 3 of float` default: **None**

### postproc.blur

#### postproc.blur.kernel_size

`list of length 2 of int` default: **[3, 3]**

#### postproc.blur.sigma

`list of length 2 of float` default: **None**

### postproc.border

#### postproc.border.size

`list of length 2 of int` default: **[1, 1]**

#### postproc.border.color

`list of length 3 of float` default: **[1.0, 1.0, 1.0]**

#### postproc.clamp

`list of length 2 of float` default: **None**

### postproc.edge

#### postproc.edge.kernel_size

`list of length 2 of int` default: **[3, 3]**

#### postproc.edge.sigma

`list of length 2 of float` default: **None**

#### postproc.edge.amount

`list of length 3 of float` default: **[1.0, 1.0, 1.0]**

#### postproc.mul

`list of length 3 of float` default: **None**

#### postproc.noise

`list of length 3 of float` default: **None**

### postproc.random_rotate

#### postproc.random_rotate.degree

`list of length 2 of float` default: **[-1, 1]**

#### postproc.random_rotate.center

`list of length 2 of float` default: **[0.5, 0.5]**

#### postproc.random_scale

`list of length 2 of float` default: **None**

#### postproc.random_shift

`list of length 2 of float` default: **None**

#### postproc.random_translate

`list of length 2 of float` default: **None**

### postproc.rotate

#### postproc.rotate.degree

`float` default: **None**

#### postproc.rotate.center

`list of length 2 of float` default: **[0.5, 0.5]**

#### postproc.shift

`list of length 2 of float` default: **None**


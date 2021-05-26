# Reference

#### `verbose`

`int` default: **`2`**

Verbosity level
- `0` = off
- `1` = show progress
- `2` = show statistics

#### `output`

`str` default: **`./`**

Directory or filename of the output. 
- If a directory, it must end with `/`. 
  In that case, the filename will be the name of the yaml config file. 
- If a filename, it must end with `.png`. Note that a number is attached to the 
  filename or is automatically increased, if the file already exists.

#### `snapshot_interval`

`int, float` default: **`20.0`**

Interval after which a snapshot of the currently trained image is saved. 

A float number specifies the interval in seconds. An integer number specifies 
the interval in number-of-epochs.

#### `epochs`

`int` default: **`300`**

The number of training steps before stopping the training, not including batch sizes. 

For example, if the number of epochs is `100` and a target has a batch_size of `10`, 
then `1000` training steps will be performed.

#### `start_epoch`

`int` default: **`0`**

The number of epochs to skip in the beginning. 

This is used by the GUI application to continue training after config changes.

#### `resolution`

`list of length 2 of int` default: **`[224, 224]`**


expression variables: [time](expressions.md#time-variables)

Resolution of the image to create. A single number for square images or two 
numbers for width and height.

It supports expression variables so you can actually change the resolution
during training, e.g:
```yaml
resolution:
- 224 if t < .2 else 448
```
would change the resolution from 224x224 to 448x448 at 20% of training time.

#### `model`

`str` default: **`ViT-B/32`**

The pre-trained [CLIP](https://github.com/openai/CLIP/) model to use. Options are `RN50`, `RN101`, `RN50x4`, `ViT-B/32`

The models are downloaded from `openaipublic.azureedge.net` and stored in the user's `~/.cache/` directory

#### `device`

`str` default: **`auto`**

The device to run the training on. Can be `cpu`, `cuda`, `cuda:1` etc.

#### `learnrate`

`float` default: **`1.0`**


expression variables: [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The learning rate of the optimizer. 

Different optimizers have different learning rates that work well. 
However, this value is scaled *by hand* so that `1.0` translates to 
about the same learning rate for each optimizer. 

The learnrate value is available to other expressions as `lr` or `learnrate`.

#### `learnrate_scale`

`float` default: **`1.0`**


expression variables: [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

A scaling parameter for the actual learning rate.

It's for convenience in the case when learnrate_scale is an expression like `1. - t`. 
The actual learnrate can be overridden with fixed values like `2` or `3` in 
different experiments.

The learnrate_scale value is available to other expressions as `lrs` or `learnrate_scale`.

#### `optimizer`

`str` default: **`adam`**

The torch optimizer to perform the gradient descent.



---


### `init`

Defines the way, the pixels are initialized. Default is random pixels.

#### `init.resolution`

`list of length 2 of int` no default

This can alter the resolution of the noise or loaded image before
it is converted to the [resolution](#resolution) of the training image.

#### `init.mean`

`list of length 3 of float` default: **`[0.5, 0.5, 0.5]`**

The mean (brightness) of the initial pixel noise. 

Can be a single number for gray or three numbers for RGB.

#### `init.std`

`list of length 3 of float` default: **`[0.1, 0.1, 0.1]`**

The standard deviation (randomness) of the initial pixel noise. 

A single number will be copied to the RGB values.

#### `init.image`

`str` no default

A filename of an image to use as starting point.

The image will be scaled to the desired resolution if necessary.

#### `init.image_tensor`

`list` no default

A 3-dimensional matrix of pixel values in the range [0, 1]  

The layout is the same as used in 
[torchvision](https://pytorch.org/vision/stable/index.html), 
namely `[C, H, W]`, where `C` is number of colors (3), 
`H` is height and `W` is width.

This is used by the GUI application to continue training after config changes.



---


### `targets`

This is a list of *targets* that define the desired image. 

Most important are the [features](reference.md#targetsfeatures) where
texts or images are defined which get converted into [CLIP](https://github.com/openai/CLIP/)
features and then drive the image creation process.

It's possible to add additional [constraints](reference.md#targetsconstraints)
which alter image creation without using CLIP, 
e.g. the image [mean](reference.md#targetsconstraintsmean), 
[saturation](reference.md#targetsconstraintssaturation) 
or [gaussian blur](reference.md#targetsconstraintsblur).

#### `targets.active`

`bool` default: **`True`**

A boolean to turn off the target during development. 

This is just a convenience parameter. To turn of a target
during testing without deleting all the parameters, simply 
put `active: false` inside.

#### `targets.name`

`str` default: **`target`**

The name of the target. 

Currently this is just displayed in the statistics dump and has no
functionality.

#### `targets.start`

`int, float` default: **`0.0`**

Start frame of the target. The whole target is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.end`

`int, float` default: **`1.0`**

End frame of the target. The whole target is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Weight factor that is multiplied with all the weights of 
[features](reference.md#targetsfeatures)
and [constraints](reference.md#targetsconstraints).

#### `targets.batch_size`

`int` default: **`1`**

The number of image frames to process during one [epoch](reference.md#epochs). 

In machine learning the batch size is one of the important and magic hyper-parameters.
They control how many different training samples are included into one weight update.

With [CLIPig](https://github.com/defgsus/CLIPig/) we are not training a neural network or anything complicated, we just
adjust pixel colors, so different batch sizes probably do not make as much 
difference to the outcome.

However, increasing the batch size certainly reduces the overall computation time. 
E.g. you can run an experiment for 1000 epochs with batch size 1, or for 100 epochs
with a batch size of 10. The latter is much faster. Basically, you can increase 
the batch size until memory is exhausted.

#### `targets.select`

`str` default: **`all`**

Selects the way how multiple [features](reference.md#targetsfeatures) are handled.

- `all`: All feature losses (multiplied with their individual [weights](reference.md#targetsfeaturesweight)) 
  are added together.
- `best`: The [similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between the 
  features of the current image pixels and each desired feature is calculated and the 
  feature with the highest similarity is chosen to adjust the pixels in it's direction.
- `worst`: Similar to the `best` selection mode, the current similarity is calculated
  and then the worst matching feature is selected. While `best` mode will generally 
  increase the influence of one or a few features, the `worst` mode will try to increase
  the influence of all features equally.
- `mix`: All individual features are averaged together 
  (respecting their individual [weights](reference.md#targetsfeaturesweight))
  and the resulting feature is compared with the features of the current image.
  This actually works quite well!

#### `targets.feature_scale`

`str` default: **`equal`**

Adjusts the initial scaling of the similarity between training image and this feature. 

- `equal`: All factors are 1.
- `fair`: Factors are set such that the **initial frame** of the training image
  has the same similarity with each feature.



---


### `targets.features`

A list of features to drive the image creation. 

The [CLIP](https://github.com/openai/CLIP/) network is used to convert texts or images
into a 512-dimensional vector of [latent variables](https://en.wikipedia.org/wiki/Latent_variable).

In the image creation process each [target](reference.md#targets) takes a section of the current image, 
shows it to [CLIP](https://github.com/openai/CLIP/) and compares the resulting feature vector with the vector of each defined feature.

Through [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) each pixel is then 
slightly adjusted in a way that would make the [CLIP](https://github.com/openai/CLIP/) feature more similar to the defined features.

#### `targets.features.text`

`str` no default

A word, sentence or paragraph that describes the desired image contents. 

[CLIP](https://github.com/openai/CLIP/) does understand english language fairly good, also *some* phrases in other languages.

#### `targets.features.image`

`str` no default

Path or URL to an image file 
([supported formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)).

Alternatively to [text](reference.md#targetsfeaturestext) an image can be converted into the
[target feature](reference.md#targetsfeatures). 

Currently the image is **resized to 224x224, ignoring the aspect-ratio** 
to fit into the [CLIP](https://github.com/openai/CLIP/) input window.

If the path starts with `http://` or `https://` it's treated as an URL and the image 
is downloaded and cached in `~/.cache/img/<md5-hash-of-url>`.

#### `targets.features.start`

`int, float` default: **`0.0`**

Start frame of the specific feature

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.features.end`

`int, float` default: **`1.0`**

End frame of the specific feature

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.features.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target feature](expressions.md#target-feature-variables), [time](expressions.md#time-variables)

A weight parameter to control the influence of a specific feature of a target.

Note that you can use negative weights as well which translates roughly to:
Generate an image that is the least likely to that feature.

#### `targets.features.scale`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target feature](expressions.md#target-feature-variables), [time](expressions.md#time-variables)

A scaling parameter that is multiplied with the **similarity** value to
yield the actual similarity used, e.g., for **best_match** [select](#targetsselect).

#### `targets.features.loss`

`str` default: **`cosine`**

The [loss function](https://en.wikipedia.org/wiki/Loss_function) used to calculate the 
difference (or error) between current and desired [feature](reference.md#targetsfeatures).

- `cosine`: The loss function is `1 - cosine_similarity(current, target)`.
  The [CLIP](https://github.com/openai/CLIP/) network was trained using 
  [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) 
  so that is the default setting.
- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.



---


### `targets.transforms`

Transforms shape the area of the trained image before showing
it to [CLIP](https://github.com/openai/CLIP/) for evaluation.

### `targets.transforms.add`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adds a fixed value to all pixels.

Three numbers specify **red**, **green** and **blue** while a 
single number specifies a gray-scale color.

### `targets.transforms.blur`

A [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) is applied to the pixels.

See [torchvision gaussian_blur](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.gaussian_blur).

#### `targets.transforms.blur.kernel_size`

`list of length 2 of int` default: **`[3, 3]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The size of the pixel window. Must be an **odd**, **positive** integer. 

Two numbers define **width** and **height** separately.

#### `targets.transforms.blur.sigma`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Gaussian kernel standard deviation. The larger, the more *blurry*.

If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.

Two numbers define sigma for **x** and **y** separately.

### `targets.transforms.border`

Draws a border on the edge of the image. The resolution is not changed.

#### `targets.transforms.border.size`

`list of length 2 of int` default: **`[1, 1]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

One integer two specify **width** and **height** at the same time, 
or two integers to specify them separately.

#### `targets.transforms.border.color`

`list of length 3 of float` default: **`[0.0, 0.0, 0.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The color of the border as float numbers in the range `[0, 1]`.

Three numbers for **red**, **green** and **blue** or a single number 
to specify a gray-scale.

### `targets.transforms.center_crop`

`list of length 2 of int` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Crops an image of the given resolution from the center.

See [torchvision center_crop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.center_crop).

One integer for square images, two numbers to specify **width** and **height**.

### `targets.transforms.clamp`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Clamps the pixels into a fixed range.

First number is the minimum allowed value for all color channels, 
second is the maximum allowed value.

An image displayed on screen or converted to a file does only include
values in the range of `[0, 1]`.

### `targets.transforms.crop`

`list of length 4 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Crops a specified section from the image.

4 numbers: **x** and **y** of top-left corner followed by **width** and **height**.

A number between 0 and 1 is considered a fraction of the full resolution.
A number greater or equal to 1 is considered a pixel coordinate

### `targets.transforms.edge`

This removes everything except edges and generally has a bad effect on image
quality. It might be useful, however.

A [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) is used to detect the edges:

    edge = amount * abs(image - blur(image))

#### `targets.transforms.edge.kernel_size`

`list of length 2 of int` default: **`[3, 3]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The size of the pixel window used for [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur). 
Must be an **odd**, **positive** integer. 

Two numbers define **width** and **height** separately.

#### `targets.transforms.edge.sigma`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Gaussian kernel standard deviation. The larger, the more *blurry*.

If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.

Two numbers define sigma for **x** and **y** separately.

#### `targets.transforms.edge.amount`

`list of length 3 of float` default: **`[1.0, 1.0, 1.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

A multiplier for the edge value. Three numbers to specify 
**red**, **green** and **blue** separately.

### `targets.transforms.fnoise`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adds noise to the image's fourier space.

It's just a bit different than the normal [noise](reference.md#targetstransformsnoise).

The noise has a scalable normal distribution around zero.

Specifies the standard deviation of the noise distribution. 
The actual value is multiplied by `15.0` to give a visually 
similar distribution as the normal [noise](reference.md#targetstransformsnoise).

One value or three values to specify **red**, **green** and **blue** separately.

### `targets.transforms.mean`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adjust the mean color value.

Three numbers specify **red**, **green** and **blue** while a 
single number specifies a gray-scale color.

### `targets.transforms.mul`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Multiplies all pixels by a fixed value.

Three numbers specify **red**, **green** and **blue** while a 
single number specifies a gray-scale color.

### `targets.transforms.noise`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adds noise to the image.

The noise has a scalable normal distribution around zero.

Specifies the standard deviation of the noise distribution. 

One value or three values to specify **red**, **green** and **blue** separately.

### `targets.transforms.pad`

Pads the image with additional pixels at the borders.

#### `targets.transforms.pad.size`

`list of length 2 of int` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The number of columns/rows to add. 

One integer to specify **x** and **y** at the same time, 
or two integers to specify them separately.

E.g. `1, 2` would add 1 column left and one column right of
the image and two rows on top and bottom respectively.

#### `targets.transforms.pad.color`

`list of length 3 of float` default: **`[0.0, 0.0, 0.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The color of the pixels that are padded around the image.

#### `targets.transforms.pad.mode`

`str` default: **`fill`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The way the padded area is filled.

- `fill`: fills everything with the `color` value
- `edge`: repeats the edge pixels
- `wrap`: repeats the image from the opposite edge

### `targets.transforms.random_crop`

`list of length 2 of int` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Crops a section of the specified resolution from a random position in the image.

See [torchvision random_crop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.random_crop)

One integer for square images, two numbers to specify **width** and **height**.

### `targets.transforms.random_rotate`

Randomly rotates the image.

Degree and center of rotation are chosen randomly between in the range
of the specified values.

The resolution is not changed and areas outside of the image
are filled with black (zero).

#### `targets.transforms.random_rotate.degree`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The minimum and maximum counter-clockwise angle of ration in degrees.

#### `targets.transforms.random_rotate.center`

`list of length 2 of float` default: **`[0.5, 0.5]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The minimum and maximum center of rotation (for x and y) in the range `[0, 1]`.

### `targets.transforms.random_scale`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Randomly scales an image in the range specified.

See [torchvision RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine).

The resolution does not change, only contents are scaled.
Areas outside of the image are filled with black (zero).

Minimum and maximum scale, where `0.5` means half and `2.0` means double.

### `targets.transforms.random_shift`

`list of length 2 or 4 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

This randomly translates the pixels of the image.

Pixels that are moved outside get attached on the other side.

Specifies the random range of translation.

A number **larger 1** or **smaller -1** translates by the actual pixels.

A number **between -1 and 1** translates by the fraction of the image resolution.
E.g., `shift: 0 1` would randomly translate the image to every possible position
given it's resolution.

**Two numbers** specify minimum and maximum shift both axis, 
**four numbers** specify minimum and maximum shift for axis **x** and **y**
respectively.

### `targets.transforms.random_translate`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Randomly translates an image in the specified range.

The resolution does not change.
Areas outside of the image are filled with black (zero).

See [torchvision RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine).

Maximum absolute fraction for horizontal and vertical translations. 
For example: `random_translate: a, b`, then horizontal shift is randomly sampled in 
the range `-img_width * a < dx < img_width * a` and vertical shift is randomly sampled in the range 
`-img_height * b < dy < img_height * b`.

### `targets.transforms.repeat`

`list of length 2 of int` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Repeats the image a number of times in the right and bottom direction.

One integer to specify **x** and **y** at the same time, 
or two integers to specify them separately.

### `targets.transforms.resize`

`list of length 2 of int` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The resolution of the image is changed.

One integer for square images, two numbers to specify **width** and **height**.

### `targets.transforms.rnoise`

Adds noise with a different resolution to the image.

The noise has a scalable normal distribution around zero.

#### `targets.transforms.rnoise.std`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Specifies the standard deviation of the noise distribution. 

One value or three values to specify **red**, **green** and **blue** separately.

#### `targets.transforms.rnoise.resolution`

`list of length 2 of int` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The resolution of the noise image. It will be 
resized to the processed image.

### `targets.transforms.rotate`

Rotates the image.

The resolution is not changed and areas outside of the image
are filled with black (zero).

#### `targets.transforms.rotate.degree`

`float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The counter-clockwise angle of ration in degrees (`[0, 360]`).

#### `targets.transforms.rotate.center`

`list of length 2 of float` default: **`[0.5, 0.5]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The center of rotation in the range `[0, 1]`. 

Two numbers to specify **x** and **y** separately.

### `targets.transforms.shift`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

This translates the image while wrapping the edges around.

Pixels that are moved outside get attached on the other side.

A number **larger 1** or **smaller -1** translates by the actual pixels.

A number **between -1 and 1** translates by the fraction of the image resolution.
E.g., `shift: .5` would move the center of the image to the previous bottom-right
corner.  

A single number specifies translation on both **x** and **y** axes while
two numbers specify them separately.



---


### `postproc`

A list of post-processing effects that are applied every epoch and change
the image pixels directly without interfering with the
backpropagation stage. 

All [transforms](transforms.md) that do not change the resolution are 
available as post processing effects.

#### `postproc.active`

`bool` default: **`True`**

A boolean to turn of the post-processing stage during development. 

This is just a convenience parameter. To turn of a stage
during testing without deleting all the parameters, simply 
put `active: false` inside.

#### `postproc.start`

`int, float` default: **`0.0`**

Start frame for the post-processing stage. The stage is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `postproc.end`

`int, float` default: **`1.0`**

End frame for the post-processing stage. The stage is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `postproc.add`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adds a fixed value to all pixels.

Three numbers specify **red**, **green** and **blue** while a 
single number specifies a gray-scale color.



---


### `postproc.blur`

A [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) is applied to the pixels.

See [torchvision gaussian_blur](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.gaussian_blur).

#### `postproc.blur.kernel_size`

`list of length 2 of int` default: **`[3, 3]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The size of the pixel window. Must be an **odd**, **positive** integer. 

Two numbers define **width** and **height** separately.

#### `postproc.blur.sigma`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Gaussian kernel standard deviation. The larger, the more *blurry*.

If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.

Two numbers define sigma for **x** and **y** separately.



---


### `postproc.border`

Draws a border on the edge of the image. The resolution is not changed.

#### `postproc.border.size`

`list of length 2 of int` default: **`[1, 1]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

One integer two specify **width** and **height** at the same time, 
or two integers to specify them separately.

#### `postproc.border.color`

`list of length 3 of float` default: **`[0.0, 0.0, 0.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The color of the border as float numbers in the range `[0, 1]`.

Three numbers for **red**, **green** and **blue** or a single number 
to specify a gray-scale.

#### `postproc.clamp`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Clamps the pixels into a fixed range.

First number is the minimum allowed value for all color channels, 
second is the maximum allowed value.

An image displayed on screen or converted to a file does only include
values in the range of `[0, 1]`.



---


### `postproc.edge`

This removes everything except edges and generally has a bad effect on image
quality. It might be useful, however.

A [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) is used to detect the edges:

    edge = amount * abs(image - blur(image))

#### `postproc.edge.kernel_size`

`list of length 2 of int` default: **`[3, 3]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The size of the pixel window used for [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur). 
Must be an **odd**, **positive** integer. 

Two numbers define **width** and **height** separately.

#### `postproc.edge.sigma`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Gaussian kernel standard deviation. The larger, the more *blurry*.

If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.

Two numbers define sigma for **x** and **y** separately.

#### `postproc.edge.amount`

`list of length 3 of float` default: **`[1.0, 1.0, 1.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

A multiplier for the edge value. Three numbers to specify 
**red**, **green** and **blue** separately.

#### `postproc.fnoise`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adds noise to the image's fourier space.

It's just a bit different than the normal [noise](reference.md#targetstransformsnoise).

The noise has a scalable normal distribution around zero.

Specifies the standard deviation of the noise distribution. 
The actual value is multiplied by `15.0` to give a visually 
similar distribution as the normal [noise](reference.md#targetstransformsnoise).

One value or three values to specify **red**, **green** and **blue** separately.

#### `postproc.mean`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adjust the mean color value.

Three numbers specify **red**, **green** and **blue** while a 
single number specifies a gray-scale color.

#### `postproc.mul`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Multiplies all pixels by a fixed value.

Three numbers specify **red**, **green** and **blue** while a 
single number specifies a gray-scale color.

#### `postproc.noise`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Adds noise to the image.

The noise has a scalable normal distribution around zero.

Specifies the standard deviation of the noise distribution. 

One value or three values to specify **red**, **green** and **blue** separately.



---


### `postproc.random_rotate`

Randomly rotates the image.

Degree and center of rotation are chosen randomly between in the range
of the specified values.

The resolution is not changed and areas outside of the image
are filled with black (zero).

#### `postproc.random_rotate.degree`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The minimum and maximum counter-clockwise angle of ration in degrees.

#### `postproc.random_rotate.center`

`list of length 2 of float` default: **`[0.5, 0.5]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The minimum and maximum center of rotation (for x and y) in the range `[0, 1]`.

#### `postproc.random_scale`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Randomly scales an image in the range specified.

See [torchvision RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine).

The resolution does not change, only contents are scaled.
Areas outside of the image are filled with black (zero).

Minimum and maximum scale, where `0.5` means half and `2.0` means double.

#### `postproc.random_shift`

`list of length 2 or 4 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

This randomly translates the pixels of the image.

Pixels that are moved outside get attached on the other side.

Specifies the random range of translation.

A number **larger 1** or **smaller -1** translates by the actual pixels.

A number **between -1 and 1** translates by the fraction of the image resolution.
E.g., `shift: 0 1` would randomly translate the image to every possible position
given it's resolution.

**Two numbers** specify minimum and maximum shift both axis, 
**four numbers** specify minimum and maximum shift for axis **x** and **y**
respectively.

#### `postproc.random_translate`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Randomly translates an image in the specified range.

The resolution does not change.
Areas outside of the image are filled with black (zero).

See [torchvision RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine).

Maximum absolute fraction for horizontal and vertical translations. 
For example: `random_translate: a, b`, then horizontal shift is randomly sampled in 
the range `-img_width * a < dx < img_width * a` and vertical shift is randomly sampled in the range 
`-img_height * b < dy < img_height * b`.



---


### `postproc.rnoise`

Adds noise with a different resolution to the image.

The noise has a scalable normal distribution around zero.

#### `postproc.rnoise.std`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

Specifies the standard deviation of the noise distribution. 

One value or three values to specify **red**, **green** and **blue** separately.

#### `postproc.rnoise.resolution`

`list of length 2 of int` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The resolution of the noise image. It will be 
resized to the processed image.



---


### `postproc.rotate`

Rotates the image.

The resolution is not changed and areas outside of the image
are filled with black (zero).

#### `postproc.rotate.degree`

`float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The counter-clockwise angle of ration in degrees (`[0, 360]`).

#### `postproc.rotate.center`

`list of length 2 of float` default: **`[0.5, 0.5]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

The center of rotation in the range `[0, 1]`. 

Two numbers to specify **x** and **y** separately.

#### `postproc.shift`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [time](expressions.md#time-variables)

This translates the image while wrapping the edges around.

Pixels that are moved outside get attached on the other side.

A number **larger 1** or **smaller -1** translates by the actual pixels.

A number **between -1 and 1** translates by the fraction of the image resolution.
E.g., `shift: .5` would move the center of the image to the previous bottom-right
corner.  

A single number specifies translation on both **x** and **y** axes while
two numbers specify them separately.



---


### `targets.constraints`

Constraints do influence the trained image without using [CLIP](https://github.com/openai/CLIP/).

They only affect the pixels that are processed by
the [transforms](transforms.md) of the [target](reference.md#targets).

### `targets.constraints.blur`

Adds the difference between the image and a blurred version to
the training loss.

This is much more helpful than using the [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur)
as a [post-processing](#postproc) step. When added to the
training loss, the blurring keeps in balance with the
actual image creation.

Areas that [CLIP](https://github.com/openai/CLIP/) is *excited about* will be constantly
updated and will stand out of the blur, while *unexciting*
areas get blurred a lot.

#### `targets.constraints.blur.kernel_size`

`list of length 2 of int` default: **`[3, 3]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The size of the pixel window. Must be an **odd**, **positive** integer. 

Two numbers define **width** and **height** separately.

#### `targets.constraints.blur.sigma`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Gaussian kernel standard deviation. The larger, the more *blurry*.

If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.

Two numbers define sigma for **x** and **y** separately.

#### `targets.constraints.blur.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.blur.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.blur.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.blur.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

### `targets.constraints.border`

Adds a border with a specific size and color to the training loss.

#### `targets.constraints.border.size`

`list of length 2 of int` default: **`[1, 1]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

One integer two specify **width** and **height** at the same time, 
or two integers to specify them separately.

#### `targets.constraints.border.color`

`list of length 3 of float` default: **`[0.0, 0.0, 0.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The color of the border as float numbers in the range `[0, 1]`.

Three numbers for **red**, **green** and **blue** or a single number 
to specify a gray-scale.

#### `targets.constraints.border.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.border.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.border.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.border.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

### `targets.constraints.contrast`

Pushes the contrast above or below a threshold value.

The contrast is currently calculated in the following way:

The image pixels are divided into the ones that are
above and below the pixel mean values. The contrast
value is then the difference between the mean of the lower
and the mean of the higher pixels.

#### `targets.constraints.contrast.above`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
below the `above` value.

#### `targets.constraints.contrast.below`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
above the `below` value.

#### `targets.constraints.contrast.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.contrast.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.contrast.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.contrast.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

### `targets.constraints.edge_mean`

Adds the difference between the current image and
and an edge-detected version to the training loss.

A [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) is used to detect the edges:

    edge = amount * abs(image - blur(image))

#### `targets.constraints.edge_mean.above`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
below the `above` value.

#### `targets.constraints.edge_mean.below`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
above the `below` value.

#### `targets.constraints.edge_mean.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.edge_mean.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.edge_mean.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.edge_mean.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

#### `targets.constraints.edge_mean.kernel_size`

`list of length 2 of int` default: **`[3, 3]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The size of the pixel window of the [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur). 
Must be an **odd**, **positive** integer. 

Two numbers define **width** and **height** separately.

#### `targets.constraints.edge_mean.sigma`

`list of length 2 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Gaussian kernel standard deviation. The larger, the more *blurry*.

If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.

Two numbers define sigma for **x** and **y** separately.

### `targets.constraints.mean`

Pushes the image color mean above or below a threshold value.

#### `targets.constraints.mean.above`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
below the `above` value.

#### `targets.constraints.mean.below`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
above the `below` value.

#### `targets.constraints.mean.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.mean.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.mean.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.mean.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

### `targets.constraints.noise`

Adds the difference between the current image and
a noisy image to the training loss.

#### `targets.constraints.noise.std`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Specifies the standard deviation of the noise distribution. 

One value or three values to specify **red**, **green** and **blue** separately.

#### `targets.constraints.noise.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.noise.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.noise.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.noise.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

### `targets.constraints.normalize`

Adds image normalization to the training loss.

The normalized version is found by moving the image colors
into the range of [min](#targetsconstraintsnormalizemin)
and [max](#targetsconstraintsnormalizemax).

#### `targets.constraints.normalize.min`

`list of length 3 of float` default: **`[0.0, 0.0, 0.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The desired lowest value in the image. 

One color for gray-scale, three colors for **red**, **green** and **blue**.

#### `targets.constraints.normalize.max`

`list of length 3 of float` default: **`[1.0, 1.0, 1.0]`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The desired highest value in the image. 

One color for gray-scale, three colors for **red**, **green** and **blue**.

#### `targets.constraints.normalize.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.normalize.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.normalize.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.normalize.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

### `targets.constraints.saturation`

Pushes the saturation above or below a threshold value.

The saturation is currently calculated as the difference of each
color channel to the mean of all channels.

#### `targets.constraints.saturation.above`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
below the `above` value.

#### `targets.constraints.saturation.below`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
above the `below` value.

#### `targets.constraints.saturation.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.saturation.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.saturation.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.saturation.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.

### `targets.constraints.std`

Pushes the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)
above or below a threshold value.

#### `targets.constraints.std.above`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
below the `above` value.

#### `targets.constraints.std.below`

`list of length 3 of float` no default


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

If specified, the training loss increases if the current value is
above the `below` value.

#### `targets.constraints.std.weight`

`float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

A multiplier for the resulting loss value of the constraint.

#### `targets.constraints.std.start`

`int, float` default: **`0.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

Start frame of the constraints. The constraint is inactive before this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.std.end`

`int, float` default: **`1.0`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

End frame of the constraints. The constraint is inactive after this time.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

- an `int` number defines the time as epoch frame
- a `float` number defines the time as ratio between 0.0 and 1.0, 
  where 1.0 is the final epoch.
- `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs.

#### `targets.constraints.std.loss`

`str` default: **`l2`**


expression variables: [learnrate](expressions.md#learnrate-variables), [resolution](expressions.md#resolution-variables), [target constraint](expressions.md#target-constraint-variables), [time](expressions.md#time-variables)

The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
used to calculate the difference (or error) between current and desired 
image.

- `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
  is the mean of the absolute difference of each vector variable.
- `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
  is the mean of the squared difference of each vector variable. Compared to 
  *mean absolute error*, it produces a smaller loss for small differences 
  (below 1.0) and a larger loss for large differences.


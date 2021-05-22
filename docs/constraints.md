# Constraints 

Constraints do influence the trained image without using [CLIP](https://github.com/openai/CLIP/).
E.g., the image [mean](reference.md#targetsconstraintsmean) can be trained
to be above or below a specific threshold. 

Constraints only affect the pixels that are processed by
the [transforms](reference.md#targetstransforms) of the [target](reference.md#targets). 

Here's a list of all available constraints:

- [blur](reference.md#targetsconstraintsblur): Adds the difference between the image and a blurred version to
    the training loss.
- [border](reference.md#targetsconstraintsborder): Adds a border with a specific size and color to the training loss.
- [contrast](reference.md#targetsconstraintscontrast): Pushes the contrast above or below a threshold value.
- [edge_mean](reference.md#targetsconstraintsedge_mean): Adds the difference between the current image and
    and an edge-detected version to the training loss.
- [mean](reference.md#targetsconstraintsmean): Pushes the image color mean above or below a threshold value.
- [noise](reference.md#targetsconstraintsnoise): Adds the difference between the current image and
    a noisy image to the training loss.
- [normalize](reference.md#targetsconstraintsnormalize): Adds image normalization to the training loss.
- [saturation](reference.md#targetsconstraintssaturation): Pushes the saturation above or below a threshold value.
- [std](reference.md#targetsconstraintsstd): Pushes the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)
    above or below a threshold value.

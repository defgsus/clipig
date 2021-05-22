# Constraints 

Constraints do influence the trained image without using [CLIP](https://github.com/openai/CLIP/).
E.g., the image [mean](#reference.mdtargetsconstraintsmean) can be trained
to be above or below a specific threshold. 

Constraints only affect the pixels that are processed by
the [transforms](reference.md#transforms) of the [target](reference.md#targets). 

Here's a list of all available constraints:

- [blur](#targetsconstraintsblur): Adds the difference between the image and a blurred version to
    the training loss.
- [border](#targetsconstraintsborder): Adds a border with a specific size and color to the training loss.
- [contrast](#targetsconstraintscontrast): Pushes the contrast above or below a threshold value.
- [edge_mean](#targetsconstraintsedge_mean): Adds the difference between the current image and
    and an edge-detected version to the training constraint.
- [mean](#targetsconstraintsmean): Pushes the image color mean above or below a threshold value
- [noise](#targetsconstraintsnoise): Adds the difference between the current image and
    a noisy image to the training loss.
- [normalize](#targetsconstraintsnormalize): Adds image normalization to the training loss.
- [saturation](#targetsconstraintssaturation): Pushes the saturation above or below a threshold value.
- [std](#targetsconstraintsstd): Pushes the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)
    above or below a threshold value.

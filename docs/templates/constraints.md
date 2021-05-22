# Constraints 

Constraints do influence the trained image without using CLIP.
E.g., the image [mean](reference.md#targetsconstraintsmean) can be trained
to be above or below a specific threshold. 

Constraints only affect the pixels that are processed by
the [transforms](reference.md#targetstransforms) of the [target](reference.md#targets). 

Here's a list of all available constraints:

{{constraints}}

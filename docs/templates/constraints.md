# Constraints 

Constraints do influence the trained image without using CLIP.
E.g., the image [mean](#targetsconstraintsmean) can be trained
to be above or below a specific threshold. 

Constraints only affect the pixels that are processed by
the [transforms](#transforms) of the [target](#targets). 

Here's a list of all available constraints:

{{constraints}}

# Intro

CLIPig generates images by using the CLIP network as an art critique. 

A bunch of pixels is continuously adjusted to increase the 
similarity of their [features](reference.md#targetsfeatures) with some user-defined
target features. Both features are derived via CLIP.

Through [backpropagation](https://en.wikipedia.org/wiki/Backpropagation),
the most common method of training artificial neural networks, the
dissimilarity of trained features and target features is 
translated back into pixel values which adjust the initial bunch of pixels
just slightly. If we do this long enough, with some
artistic variation in the processing pipeline, an actual image emerges.  

CLIPig is designed to allow a lot of control over those *variations* 
which requires a bit of documentation.

Please browse through the *walk-through* to get an overview and follow 
the links to the reference pages and any point.

- [Walk-through](walkthrough.md)
- [Command line interface](cli.md)
- [Expressions](expressions.md)
- [Transforms](transforms.md)
- [Constraints](constraints.md)
- [Parameter reference](reference.md)

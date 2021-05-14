# CLIPig documentation

CLIPig generates images by using the CLIP network as an art critique. 

A bunch of pixels is continuously adjusted to increase the 
similarity of their [features](#targetsfeatures) with some user-defined
target features. Both features are derived via CLIP.

Through [backpropagation](https://en.wikipedia.org/wiki/Backpropagation),
the most common method of training artificial neuronal networks, the
dissimilarity of trained features and target features is 
translated back into pixel values which adjust the initial bunch of pixels
just slightly. If we do this long enough, with some
artistic variation in the processing pipeline, an actual image derives.  

CLIPig is designed to allow a lot of control over those *variations* 
which requires a bit of documentation. 

- [introduction](#introduction)
- [command line tool](#??)
- [transforms](#transforms)
- [constraints](#constraints)
- [parameter reference](#reference)


## Introduction

First of all, experiments are defined in [YAML](https://yaml.org/) files.
I actually prefer [JSON](https://www.json.org/) but it does not support
comments out of the box and is quite strict with those trailing commas..
Anyways, the basic desires of defining lists and key/value maps are indeed
quite human-friendly in YAML: 

```yaml
a_list:
  - first entry
  - second entry

a_map:
  first_key: first value
  second_key: second value

# a comment
```

And that's all to know about YAML for our purposes. 

Now in CLIPig the *desire* for an image is expressed as a [target](#targets).
There can be multiple targets and each target can have multiple 
[target features](#targetsfeatures).

```yaml
targets:
  - features:
      - text: a curly spoon
```

For a live experience in image generation call

```shell script
python clipig-gui.py
```

paste the code inside the editor and press `Alt-S` to start training 
and watch the image emerge in realtime.

So, what does the image look like?

![a badly rendered curly spoon](demo1.png)

Yeah, well... I promised *images* and now i'm showing nothing more than 
a psychedelic pixel mess. 

But indeed, CLIP does think this image to be **95%** similar 
to the term **a curly spoon**. This is a top scoring that
an actual photo never would get and a classic example of an 
[Adversarial](https://en.wikipedia.org/wiki/Adversarial_machine_learning)
in machine learning. 

To make it look more like an actual image add some of those 
artistic variations, spoken of earlier. The art is in showing different
parts of the image to CLIP when evaluating the feature similarities. 

This is accomplished via [transforms](#transforms):

```yaml
targets:
  - features:
      - text: a curly spoon
  transforms:
    - random_shift: 0 1
```

![a slightly better rendered curly spoon](demo2.png)

The [random_shift](#targetstransformsrandom_shift) transformation
simply moves the image center to a random position, before
each evaluation by CLIP. The edges are wrapped around so the
outcome is actually a repeatable texture! The object of interest
might just not be in it's center. 

Apropos, the object of interest might look a bit spoony to a
human observer but not so much, i'd say. There is a lot of curliness
in the background but the spoon does not show as much. CLIP also
seems to lack curliness of the spoon because actual letters appeared
to increase the similarity nevertheless. It got to **50%**. 

Another method to inspire CLIP is 
[random rotation](#targetstransformsrandom_rotation). 

```yaml
targets:
  - features:
      - text: a curly spoon
  transforms:
    - random_rotate:
        degree: -90 90
        center: 0.3 .7
```

![a slightly better rendered curly spoon](demo3.png)

Each evaluated image is first rotated randomly between -90 and +90 
degree with a random center in the middle 2/3rds of the image. This
does not create a repeatable texture and the edges are typically 
a bit underdeveloped because they get rotated out of the visible area
for a lot of times. 

It shows some good areas with shiny metal and spoony curliness but
it's not recognizable as a spoon too much.


Let's go ahead and add some other stuff:

```yaml
targets:
- batch_size: 5
  features:
  - text: a curly spoon on a plate
  transforms:
  - noise: 0.1*ti
  - random_shift: -.1 .1
  - random_rotate:
      degree: -3 3
      center: 0.3 .7
  constraints:
  - blur: 
      kernel_size: 31*ti
```

![a not so bad rendered curly spoon](demo4.png)

First the number of evaluations was increased 
with the [batch size](#targetsbatch_size) parameter. 
That results in a runtime of about 2 minutes on 1500 cuda cores.

`on a plate` was added to the target text to make CLIP somewhat more 
opinionated about the background.

Some [noise](#targetstransformsnoise) is added to each image that is
shown to CLIP and a [gaussian blur](#targetsconstraintsblur) is added
to the backpropagation [loss](https://en.wikipedia.org/wiki/Loss_function).  

The noise makes CLIPig kind of *think twice* about the the idea of adjusting
a pixel's color. The blur used as a training loss tends to blur out the
areas where CLIP is not interested in, while the points of interest are
constantly updated and are not blurred as much. Unfortunately both
methods also help to create other artifacts. And this is where those
*variations* start to become *artistic*. It certainly takes some patience.

And maybe the correct language. What if we change the target text to
`a photo of a curly spoon`?

![almost a photo of a curly spoon](demo5.png)

Ah, i see what CLIP is thinking there.

## command line interface

A yaml file is rendered via

```shell script
python clipig.py demo.yaml -o images/
```

to produce a [PNG](https://en.wikipedia.org/wiki/Portable_Network_Graphics)
file at `images/demo.png`. When called again, `images/demo-1.png` will be 
created, `images/demo-2.png` the third time, aso.. 

Generally, `clipig.py` never overwrites existing files. 

The `-o` (or `--output`) option is not mandatory but it's good practice
to store those images in separate directories because they tend to grow
in number.

You can specify an actual filename with `-o /path/image.png`, otherwise
the name of the yaml file is used. Still, if `/path/image.png` already
exists, `/path/image-1.png` will be created.

CLIPig also stores a `<filename>.yaml` file besides the image, if there
does not exist one already, which holds the complete configuration with 
all defaults and the runtime in seconds as comment on the top. 


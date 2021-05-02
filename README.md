# (yet another) CLIP Image Generator

Dear seeker!

This is yet another pixel back-propagation engine using OpenAI's 
[CLIP](https://github.com/openai/CLIP/) network to rate the validity.

It uses no sophisticated image generation network, just ordinary RGB pixel planes.

The outstanding thing, maybe, and the reason for developing it, is it's configuration interface.
I got pretty tired of constantly changing actual code during various experiments so i started
this new project which fulfills *most* desires through yaml configuration files. 


### example yaml config

```yaml
resolution: 1024
epochs: 2000

learnrate: 1.5
learnrate_scale: (1. - .95 * pow(t, 5.))   # 't' is training epoch in [0, 1] range

targets:

  - name: random sampler

    features:
      - text: some text feature to match
      - text: some text feature to avoid
        weight: -1.
      - image: /path/to/image.png
        loss: L2  # 'cosine' is default, 'L!' and 'L2' are also possible

    transforms:
      - random_scale: .1 1.
      - random_crop: 224

  - name: image adjustments
    end: 30%
    constraints:
      - mean:
          above: .1 .2 .3

# post-processing is applied after each back-propagation step
postproc:
  - blur: 3 .35
    end: 50%    
```

As you can see, it supports 
- expressions
- multiple targets
- multiple features per target
- CLIP features from texts or images
- negative weights!
- arbitrary pixel transformation chains
- a couple of other constraints that can be added to the loss
- a couple of image post-processing effects 
- scheduling via `start` and `end` parameters

---

Once the interface is settled i'll write down the documentation. 

Currently, to get started switch to a virtual environment that contains 
the `torch` library matching your CUDA drivers and then

```bash
python clipig.py examples/strawberries.yaml -o ./images/
```

--- 

Here's a motivating [article](https://defgsus.github.io/blog/2021/04/28/malazan-clip-features.html)
whose images where created with the predecessor of this code.

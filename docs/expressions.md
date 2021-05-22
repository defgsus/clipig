# Expressions

[CLIPig](https://github.com/defgsus/CLIPig/) supports expressions for all parameters. Some parameters
also support [variables](#Expression-variables) 
and the expression will be evaluated
every time the value is needed. 

E.g., if you want thrice the [CLIP](https://github.com/openai/CLIP/)-resolution of 224x224 pixels
but have reasons to not calculate it, just say:

```yaml
resolution: 224*3
```

> **Note**: Parameters that expect lists (like **resolution** above)
> copy a single value to all entries of the list. A list can 
> be specified with 
> - YAML syntax:
>   ```yaml
>   resolution: 
>     - 640
>     - 480
>   ```
> - with commas:
>   ```yaml
>   resolution: 640, 480
>   ```
> - or simply with spaces
>   ```yaml
>   resolution: 640 480
>   ```
> If you type expressions, you might want to use spaces or 
> commas. In case of list parameters you'll need to use the 
> YAML lists:
> ```yaml
> resolution:
>   - 224 * 3
>   - pow(224, 1.2)
> ``` 


The result of an expression is automatically converted to 
the desired type. So even if your `resolution` expression 
generates a float it will be cast to integer before being used.

> **Note**: Divisions through zero and stuff like this will
> throw an error and stop the experiment.

## Expression variables

### time variables

Holds variables that reference the current training time.

#### `epoch` variable

type: `int`

The current epoch / frame, starting at zero.
#### `time` variable

type: `float`

The current epoch / frame divided by the number of epochs, or in
other words: A float ranging from **0.0** (start of training) to 
**1.0** (end of training).
#### `time_inverse` variable

type: `float`

One minus the current epoch / frame divided by the number of epochs, or in
other words: A float ranging from **1.0** (start of training) to 
**0.0** (end of training).
#### `time_step` variable

type: `function(float, float)`

A function that returns a float in the range [0, 1]
during the time interval defined by the two values.
```python
time_step(0, 1)    # increases from zero to one during whole training
time_step(0.5, 1)  # increases from zero to one during second half of training
time_step(0.5, 0)  # decreases from one to zero during first half of training
```
### resolution variables

Holds the resolution of the training image.

#### `resolution` variable

type: `[int, int]`

The resolution of the training image as list of **width** and **height**.
#### `width` variable

type: `int`

The width of the training image.
#### `height` variable

type: `int`

The width of the training image.
### learnrate variables

The current values of [learnrate](reference.md#learnrate) 
and [learnrate_scale](reference.md#learnrate_scale)
which can be expressions themselves.

#### `learnrate` variable

type: `float`

The currently used [learnrate](reference.md#learnrate)
#### `learnrate_scale` variable

type: `float`

The currently used [learnrate_scale](reference.md#learnrate_scale)
### target feature variables

Variables available to [target features](reference.md#targetsfeatures)

#### `similarity` variable

type: `float`

The [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
of the [CLIP](https://github.com/openai/CLIP/)-representation of the current, transformed image area 
with the desired feature.

The value is in the range [-100, 100].
### target constraint variables

Variables available to [constraints](reference.md#targetsconstraints)

#### `similarity` variable

type: `float`

The mean of all [cosine similarities](https://en.wikipedia.org/wiki/Cosine_similarity)
of the [CLIP](https://github.com/openai/CLIP/)-representation of the current, transformed image area 
with the desired features of this target.

The value is in the range [-100, 100].

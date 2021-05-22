# Expressions

CLIPig supports expressions for all parameters. Some parameters
also support [variables](#Expression-variables) 
and the expression will be evaluated
every time the value is needed. 

E.g., if you want thrice the CLIP-resolution of 224x224 pixels
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

{{variables}}

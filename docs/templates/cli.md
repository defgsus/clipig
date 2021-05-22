# command line interface

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

Multiple yaml files are merged into one set of parameters, e.g.:

```shell script
python clipig.py best-meta-settings.yaml specific.yaml
```

will parse `best-meta-settings.yaml` and then add anything
from `sepcific.yaml` on top. List entries 
like [targets](#targets) will be appended to the previous list.

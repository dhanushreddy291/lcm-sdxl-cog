# latent-consistency/lcm-sdxl

This is an implementation of the [latent-consistency/lcm-sdxl](https://huggingface.co/latent-consistency/lcm-sdxl) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="A beautiful brunette pilot girl, beautiful, moody lighting, best quality, full body portrait, real picture, intricate details, depth of field, in a cold snowstorm, , Fujifilm XT3, outdoors, bright day, Beautiful lighting, RAW photo, 8k uhd, film grain, unreal engine 5, ray travig"

## Example:

"A beautiful brunette pilot girl, beautiful, moody lighting, best quality, full body portrait, real picture, intricate details, depth of field, in a cold snowstorm, , Fujifilm XT3, outdoors, bright day, Beautiful lighting, RAW photo, 8k uhd, film grain, unreal engine 5, ray travig"

![A beautiful brunette pilot girl, beautiful, moody lighting, best quality, full body portrait, real picture, intricate details, depth of field, in a cold snowstorm, , Fujifilm XT3, outdoors, bright day, Beautiful lighting, RAW photo, 8k uhd, film grain, unreal engine 5, ray travig](out-0.png)
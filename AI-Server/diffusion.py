import replicate
import os

os.environ["REPLICATE_API_TOKEN"] = "20e3ac5d8867905923cf9c9e823a6a8012fe67eb"
model = replicate.models.get("jagilley/controlnet-scribble")
version = model.versions.get("435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117")
def creat_image(image, caption):

    # https://replicate.com/jagilley/controlnet-scribble/versions/435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117#input
    inputs = {
        # Input image
        'image': open(image, "rb"),

        # Prompt for the model
        # 'prompt': "cartoon, softly, disney, fairy tale, classic, ordinary",
        'prompt' : caption + ", high quality, photorealistic, sharp focus, depth of field",

        # Number of samples (higher values may OOM)
        'num_samples': "1",

        # Image resolution to be generated
        'image_resolution': "512",

        # Steps
        'ddim_steps': 20,

        # Guidance Scale
        # Range: 0.1 to 30
        'scale': 9,

        # Seed
        'seed': 0,

        # eta (DDIM)
        'eta': 0,

        # Added Prompt
        'a_prompt': "best quality, extremely detailed",

        # Negative Prompt
        'n_prompt': "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        # 'n_prompt': "",

    }

    # https://replicate.com/jagilley/controlnet-scribble/versions/435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117#output-schema
    output = version.predict(**inputs)
    return output

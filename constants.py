UNET_LAYERS = ['IN01', 'IN02', 'IN04', 'IN05', 'IN07', 'IN08', 'MID',
               'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'OUT09', 'OUT10', 'OUT11']

SD_INFERENCE_TIMESTEPS = [999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659,
                          639, 619, 599, 579, 559, 539, 519, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300,
                          280, 260, 240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20]

PROMPTS = [
    "A photo of a {}",
    "A photo of {} in the jungle",
    "A photo of {} on a beach",
    "A photo of {} in Times Square",
    "A photo of {} in the moon",
    "A painting of {} in the style of Monet",
    "Oil painting of {}",
    "A Marc Chagall painting of {}",
    "A manga drawing of {}",
    'A watercolor painting of {}',
    "A statue of {}",
    "App icon of {}",
    "A sand sculpture of {}",
    "Colorful graffiti of {}",
    "A photograph of two {} on a table",
]

VALIDATION_PROMPTS = [
    "<img_unknown>",
]

IMAGENET_TEMPLATES_SMALL = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

IMAGENET_STYLE_TEMPLATES_SMALL = [
    "a {} in the style of {}",
    "a {} inspired by {}",
    "a {} reflecting {}",
    "a {} designed after {}",
    "a {} created in the style of {}",
    "a {} with the influence of {}",
    "a {} modeled after {}",
    "a {} based on {}",
    "a {} shaped by {}",
    "a {} crafted in the style of {}",
    "a {} representing {}",
    "a {} following the style of {}",
    "a {} related to {}",
    "a {} adapted from {}",
    "a {} evoking {}",
    "a {} interpreted through {}",
    "a {} aligned with {}",
    "a {} derived from {}",
    "a {} reimagined through {}",
    "a {} styled by {}"
]

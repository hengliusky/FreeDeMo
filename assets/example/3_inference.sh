#!/bin/bash

python scripts/animate_seg.py \
    --inference_config configs/inference/inference_test.yaml \
    --n_prompt "nsfw, paintings, cartoon, anime, dark,sketches, wrong white balance,worst quality, low quality, normal quality, lowres, watermark, monochrome, grayscale, ugly, blurry, Tan skin, dark skin, black skin, skin spots, skin blemishes, age spot, glans, disabled, bad anatomy, amputation, bad proportions, twins, missing body, extra claws,fused body, extra head, poorly drawn face, bad eyes, deformed eye, unclear eyes, cross-eyed, long neck, malformed limbs, extra limbs, extra arms, missing arms, bad tongue, strange fingers, mutated hands, missing hands, poorly drawn hands, extra hands, fused hands, connected hand, bad hands, missing fingers, extra fingers, 4 fingers, 3 fingers, deformed hands, extra legs, bad legs, many legs, more than two legs, bad feet,extrafeets,incongruousappearance,Distortedappearance,deformation,bad_pictures, negative_hand-neg , bad teeth, mutated hands and fingers, bad anatomy" \
    --prompt "a girl is smiling,a boy is nodding up and down." \
    --path_to_first_frame "assets/example/003.png" \
    --format mp4 \
    --seed 75831 \
    --frame_stride 2







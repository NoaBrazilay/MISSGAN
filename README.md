# Deep Learning Project - MISS GAN

Note: The code was run in windows with GTX 1070.

Our code was built upon the StraGAN v2 framework:

[https://github.com/clovaai/stargan-v2](StarGAN-v2)

For training the MISS GAN framework:

`python main.py --mode train --num_domains 2 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --vgg_w 1`

For creating images from a reference illustration:

`python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 0 --checkpoint_dir expr\checkpoints\ --result_dir expr\results\ --src_dir assets\representative\illustrations\src --ref_dir assets\representative\illustrations\ref`

For creating images from a randomized latent code:

`python main.py --mode eval --num_domains 2 --w_hpf 0 --resume_iter 100000 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --checkpoint_dir expr\checkpoints\ --eval_dir expr\eval`


Note that this study contains code from the following repositories as well:

[https://github.com/NVlabs/MUNIT](MUNIT)

[https://github.com/giddyyupp/ganilla](GANILLA)

Copyright (c) 2020 NoaBarzilay
# Deep Learning Project - MILLGAN

Note: The code was run in windows with GTX 1070.
1. Downlad the code from the git site.
2. Download Dataset
Since the illustration dataset can't be publicly avalable (legal rights) we will not provide the drive link for the dataset here.
We shared a folder named "project_307923052_302636675" with barakhadad@mail.tau.ac.il (if you encounter any problems please contact us and we will respond immediately
noabarzilay11@gmail.com).
StarGAN framework-
In order to be able to run the code with the illustration dataset download go to:
the folder project_307923052_302636675 -> DATASET ->  StarGanDataset
And download the illustrations.zip file and extract it in the following path:
your_path_to_our_project\DeepLearningProject\stargan\data\Extract the folder here
MUNIT framework-
In order to be able to run the code with the illustration dataset download go to:
the folder project_307923052_302636675 -> DATASET ->  MUNITDataset
And download the illustrations2landscapes.zip file and extract it in the following path:
your_path_to_our_project\DeepLearningProject\MUNIT\datasets\Extract the folder here
3. Training
in order to train the different genrators fro scratch-
StarGAN framework-
In the report we described several models, we are going to elaborate here how to train each model:
MillGAN -
python main.py --mode train --num_domains 2 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --vgg_w 1
Model A- Baseline StarGan -
python main.py --mode train --num_domains 2 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --use_star_gen 1
Model B-
python main.py --mode train --num_domains 2 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --use_residual_upsample 0
Model D-
python main.py --mode train --num_domains 2 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --loss_sacl 1
Model E
python main.py --mode train --num_domains 2 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --vgg_w 1 --loss_sacl 1
MUNIT framework-
In the report we described several models, we are going to elaborate here how to train each model:
Model A - baseline-
Go to "illustrations2landscapes_folder.yaml" in MUNIT\configs-
Chane in the file the following-
ganilla_gen: False
Model B-
Chane in the file the following-
use_style_enc_simple: True
Model C-
Chane in the file the following-
use_style_enc_simple: False
Model E-
Chane in the file the following-
use_patch_gan: False
Model F-
Chane in the file the following-
style_dim: 16
For all Run the following command -
python train.py --config configs/illustrations2landscapes_folder.yaml

4. Evaluating pretrained models -
StarGAN framework-
In order to download pretrained models go to:
the folder project_307923052_302636675 -> TrainedGenerators->  StarGan
Pick the pretrained model you wish to evaluate and enter the folder (for example "MillGAN").
The dorralated pretrained models according to the report-
Model A - exprOriginal
Model B- exprGanilla_AdainNoResidualinup
Model C- exprGanilla_AdainResBlocksinUp
Model D- exprGanilla_AdainResBlocksinUp_SACL
Model E- exprGanilla_AdainResBlocksinUp_VGGContentLoss_SACL
MillGAN - it's simply MillGAN :)
All other models in the drive appears in the Appendix of the report.

copy the content of the checkpoints folder from the model you pick to :
your_path_to_our_project\DeepLearningProject\stargan\expr\checkpoints
Rum the following command -
IMPORTANT - You need to add the additional flags presented in the former section to the command line here as well (for example for evaluating the baselime model you shuld run the commands with --use_star_gen 1)
For Generate images:
python main.py --mode eval --num_domains 2 --w_hpf 0 --resume_iter 100000 --train_img_dir data\illustrations\train --val_img_dir data\illustrations\val --checkpoint_dir expr\checkpoints\ --eval_dir expr\eval
For generate images from refences illustrations:
Create a folder with the following path:
assets\representative\illustrations\src
Drop there all the natural images you wish to transfer to illustrations.
Create another folder with the folwing path:
assets\representative\illustrations\ref
Drop there all of the illustration images you wish the generator will extract the style code from.
Run the following command and you will get an image simillar to Figure 9 from report:
python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 0 --checkpoint_dir expr\checkpoints\ --result_dir expr\results\ --src_dir assets\representative\illustrations\src --ref_dir assets\representative\illustrations\ref

MUNIT framework-
In order to download pretrained models go to:
the folder project_307923052_302636675 -> TrainedGenerators->  MUNIT
Download the models direcory and copy it to:
your_path_to_our_project\DeepLearningProject\MUNIT\models
Model A - OriginalMUNIT
Model B- allup_adain_vgg_ncyc_styleEncMunit_total
Model C- allup_adain_vgg_cyc_total
Model D- allup_adain_vgg_ncyc_styleEncMunit5nd_total
Model E- allup_adain_vgg_ncyc_styleEncMunit_total_MSdis
ModelF - allup_adain_vgg_ncyc_styleEncMunit_16styledim_total
All other models in the drive appears in the Appendix of the report.
IMPORTANT - Each model you wish to test shole be tested with the proper configuration presented in the former section "Training", for example, Model A should contain in the illustrations2landscapes_folder.yaml file "ganilla_gen:  False".
python test.py --config configs/illustrations2landscapes_folder.yaml --input inputs/Input_image_you_chose --output_folder results/model_results --checkpoint models/Pre-trained_model_you_chose.pt --a2b 0
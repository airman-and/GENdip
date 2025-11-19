import os


working_dir = os.getcwd()
project_dir = working_dir[:-4] if working_dir.endswith("/src") else working_dir

model_path = f"{project_dir}/model/vae_celeba_128_learning_rate_0.0005_epoch_10_latent_dim_128.pth"
celebA_path = f"{project_dir}/dataset/celebA"
celebA_image_path = f"{celebA_path}/img_align_celeba/img_align_celeba"
celebA_attr_path = f"{celebA_path}/list_attr_celeba.csv"
output_path = f"{project_dir}/output"

model_latent_dim = 128
batch_size = 128
image_size = 128

scale = 3
shuffle = False

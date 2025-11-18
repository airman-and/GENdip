import os


working_dir = os.getcwd()
project_dir = working_dir[:-4] if working_dir.endswith("/src") else working_dir

model_path = f"{project_dir}/model/vae_celebA/vae_celeba_latent_200_epochs_10_batch_64_subset_80000.pth"
celebA_path = f"{project_dir}/dataset/celebA"
celebA_image_path = f"{celebA_path}/img_align_celeba/img_align_celeba"
celebA_attr_path = f"{celebA_path}/list_attr_celeba.csv"
output_path = f"{project_dir}/output"

scale = 3

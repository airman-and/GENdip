import os

# Get the absolute path of the directory containing this file (src/)
src_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (GenDL-LatentControl/)
project_dir = os.path.dirname(src_dir)

model_path = f"{project_dir}/model/vae_celeba_128_learning_rate_0.0005_epoch_10_latent_dim_128.pth"
aae_model_path = f"{project_dir}/checkpoints/aae_epoch_30_latent_128.pth"
celebA_path = f"{project_dir}/dataset/celebA"
celebA_image_path = f"{celebA_path}/img_align_celeba/img_align_celeba"
celebA_attr_path = f"{celebA_path}/list_attr_celeba.csv"
output_path = f"{project_dir}/output"

model_latent_dim = 128
batch_size = 128
image_size = 128

scale = 3
shuffle = False


import sys
import os
sys.path.append(os.getcwd())
from stylegan2_ada_pytorch.torch_utils import custom_ops

print("Attempting to load upfirdn2d_plugin...")
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(os.getcwd(), 'tmp_extensions')
custom_ops.verbosity = 'full'
try:
    custom_ops.get_plugin('upfirdn2d_plugin', sources=['stylegan2_ada_pytorch/torch_utils/ops/upfirdn2d.cpp', 'stylegan2_ada_pytorch/torch_utils/ops/upfirdn2d.cu'])
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")

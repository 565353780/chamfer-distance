# pip install -U torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -U icecream jax

# faiss
conda install -c conda-forge -c nvidia -c rapidsai cuvs
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.11.0

# kaolin
pip install kaolin==0.17.0 -f \
  https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

pip install -e .

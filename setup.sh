pip install -U torch torchvision torchaudio

pip install -U icecream jax

conda install -c conda-forge -c nvidia -c rapidsai cuvs
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.11.0

pip install -e .

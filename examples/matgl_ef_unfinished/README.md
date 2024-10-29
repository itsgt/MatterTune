pip install matgl
pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall dgl
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install mp_api
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.2.1%2Bcu121.html
pip install einops
pip install e3nn
pip install wandb
wget -O conda_install.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash conda_install.sh -b -p $HOME/miniconda
rm conda_install.sh
eval "$(~/miniconda/bin/conda shell.bash hook)"
pip install -r requirements.txt

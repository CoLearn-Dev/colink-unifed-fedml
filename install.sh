eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

if { conda env list | grep 'colink-protocol-unifed-fedml'; } >/dev/null 2>&1; then
    conda env remove -n colink-protocol-unifed-fedml
fi
conda create -n colink-protocol-unifed-fedml python=3.10 -y
conda activate colink-protocol-unifed-fedml

conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/FedML-AI/FedML.git
cd FedML
git checkout a8b59a6346bf548dc66bd2266af5070ee15db4eb
cd python
pip install aiohttp==3.8.1
pip install -e .

cd ../..
pip install -e .
pip install pytest pympler
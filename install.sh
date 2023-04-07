if { conda env list | grep 'colink-protocol-unifed-fedml'; } >/dev/null 2>&1; then
    conda env remove -n colink-protocol-unifed-fedml
fi
conda create -n colink-protocol-unifed-fedml python=3.10 -y
conda activate colink-protocol-unifed-fedml
pip install fedml==0.8.2
pip install -e .
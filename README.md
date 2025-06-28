


conda create -n RepMobile python=3.10 
conda activate RepMobile
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full
mim install mmdet
mim install mmseg
pip install thop
pip install coremltools==6.3
 pip install prettytable
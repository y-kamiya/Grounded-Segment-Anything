export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.7


venv_dir=.venv-gsam
if [ ! -d $venv_dir ]; then
    python -m venv $venv_dir
fi
source $venv_dir/bin/activate

# correspondance with cuda11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

python -m pip install -e segment_anything

# avoid error of not matching cuda version
# cf. https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/412#issuecomment-1873247424
rm -f GroundingDINO/pyproject.toml
python -m pip install -e GroundingDINO

pip install --upgrade "diffusers[torch]"


if [ ! -d recognize-anything ]; then
    git clone https://github.com/xinyu1205/recognize-anything.git
    # avoid syntax error
    sed -i -e "s/install_requires/#install_requires" recognize-anything/setup.cfg
fi
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/

pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

# for gradio app
pip install gradio<4.0 litellm
# avoid error of not found blip2 components
pip install transformers>=4.33.1

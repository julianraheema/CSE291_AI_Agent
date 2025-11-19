## Using ecua_omniparser_gpu.py

Here is a small tutorial for using the omniparser v2 code inside `ecua_omniparser_gpu.py`.  
First follow this code from the omniparser github for creating the conda env:  
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```
You will also (probably) update `pytorch torchvision torchaudio pytorch-cuda=12.x` to the right cuda version as well.  
Next, using a gitbash terminal, download the yolo and florence v2 weights, make sure you are in the `third_party/omniparser` folder of our repo.  
Run this code in gitbash terminal:  
```
   # download the model checkpoints to local directory OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```
### Running the Code:
I have provided a small sample code that runs omniparser: `python .\ecua_omniparser_gpu.py`  
You can also define the OmniEngine object and run one parse with this code: `engine = OmniEngine()` and `out = engine.parse_fullscreen(save_overlay=True)`

### What does Omniparser return?
I set the code up to return a dictionary with the `window` size, `elements`, the `overlay_path` of the annoted image, and the total `num_elements`.  
You can find more specifics in details at the bottom of `ecua_omniparser_gpu.py`.  
From here you should be able to disect the dictionary or convert to a json for the planner/controller.  
# <img width="372.5" height="125" alt="image" src="https://github.com/user-attachments/assets/c522e010-1377-452c-b6ca-f4e266df1385" />

An image based AI system for automated early detection of lumpy skin disease to save lives and secure rural family wealth.


## How to test images?
### Test using Google Colab
1) Upload the file `test_ConvNeXt_ONNX_Colab.ipynb` (located inside the **scripts** folder) to a Google Drive account.
2) Open the file using Google Colab.
3) Under the 3rd section, i.e., **3. Same test, using the ONNX model hosted on GitHub**, run the two code cells consecutively.

### Test using a local machine
1) Download the following files and folder to your local machine : a) `convnext_cattle_best.onnx`; <br />b) `test_onnx.py` (located inside the **scripts** directory); <br />c) **test_images** folder
2) Install a code editor like Visual Studio Code.
3) Install Python 3.
4) Install the following Python packages: json, numpy, pillow, onnxruntime.
5) Open the file `test_onnx.py`.
6) Update the file paths for __**ONNX_PATH**__ (line 19) and __**INPUT_PATHS**__ (line 24) to point to the downloaded `convnext_cattle_best.onnx` model and input files.
7) Save and run the file.

## Dataset source
https://universe.roboflow.com/operating-room/cows-mien3

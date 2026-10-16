# <img width="372.5" height="125" alt="image" src="https://github.com/user-attachments/assets/6018cc9e-8221-43dc-8130-f0576b00af10" />


An image based AI system for automated early detection of lumpy skin disease to save lives and secure rural family wealth.


## How to test images?
### Test using Google Colab
1) Upload the file `test_ConvNeXt_ONNX_Colab.ipynb` (located inside the **scripts** folder) to a Google Drive account.
2) Open the file using Google Colab.
3) Run the code cell under the 3rd section, i.e., **3. Load the ONNX model from GitHub (run once)**.
4) Under the 4th section, i.e., **4. Test images from any link or path (re-run freely)**, update the file paths __**/content/drive/MyDrive/...**__.
5) Save and run the two code cells.

### Test using a local machine
1) Download the following files and folder to your local machine : <br />a) `convnext_cattle_best.onnx` <br />b) `test_onnx.py` (located inside the **scripts** directory) <br />c) **test_images** folder
2) Install a code editor like Visual Studio Code.
3) Install Python 3.
4) Install the following Python packages: json, numpy, pillow, onnxruntime.
5) Open the file `test_onnx.py`.
6) Update the file paths for __**ONNX_PATH**__ (line 19) and __**INPUT_PATHS**__ (line 24) to point to the downloaded `convnext_cattle_best.onnx` model and input files.
7) Save and run the file.

## Dataset source
https://universe.roboflow.com/operating-room/cows-mien3

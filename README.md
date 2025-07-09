# Multi-Label Logo Recognition

This code is authored by Maria Luisa Bernabéu, Javier Gallego and Antonio Pertusa

This repository is for the paper ‘Multi-Label Logo Recognition and Retrieval based on Weighted Fusion of Neural Features’ in https://doi.org/10.1111/exsy.13627

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Prerequisites
•	Python
•	Tensorflow with NVIDIA GPU or CPU
•	OpenCV
•	numpy

## Training
For a given dataset, we train the network and save weights file (if arg load==false) into WEIGHTS folder. If arg save==true, we save NC in a characteristic vector file for train set and test set into the NC folder. To train images stored in DATA folder, you must specify the csv file with the parameter -csv and  select the characteristic type to train by the parameter -type when calling py_logos.py
**Classification type**: maincat, subcat, color, shape, text, mainsec, ae


Example:
>	python py_logos.py -csv salida.csv -img DATA -type maincat --aug –save


You can download the weights from trained models, unzip and put them into WEIGHTS/. To use the weights you can specify --load  when calling py_logos.py.

For example:
>	python py_logos.py -csv salida.csv -img DATA -type maincat --load --aug

## kNN classification
 To classify images in a folder DATA, you must  specify the NC of train file with the parameter -train, the NC of test file with the parameter -test, and select the characteristic type with the parameter -type when calling py_logos_knn.py.
 
For example:
> python py_logos_knn.py -train NC/features_type_color_train.csv -test NC/features_type_color_test.csv -csv salida.csv -img DATA -type color --v





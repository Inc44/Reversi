## Usage
```powershell
git clone https://github.com/Inc44/Reversi.git
cd Reversi
conda create --name reversi python=3.13 -y
conda activate reversi
pip install flask==3.1.0
pip install numpy==2.2.0
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -OO reversi.py --gui
color 70
python -OO reversi.py
```

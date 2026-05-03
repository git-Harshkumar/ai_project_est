@echo off
echo ============================================
echo  Installing required Python packages...
echo ============================================
pip install tensorflow scikit-learn Pillow seaborn matplotlib numpy
echo.
echo ============================================
echo  All packages installed!
echo  Now run:  python train.py
echo ============================================
pause

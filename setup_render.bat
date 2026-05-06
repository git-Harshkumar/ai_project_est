@echo off
echo ========================================
echo  Setting up Pipeline for Render Deploy
echo ========================================

REM Create directories inside Pipeline
if not exist "Pipeline\model" mkdir "Pipeline\model"
if not exist "Pipeline\reports" mkdir "Pipeline\reports"
if not exist "Pipeline\plots" mkdir "Pipeline\plots"

REM Copy model
echo Copying model...
copy /Y "Dataset\model\best_model.keras" "Pipeline\model\best_model.keras"

REM Copy report
echo Copying report...
copy /Y "Dataset\reports\classification_report.txt" "Pipeline\reports\classification_report.txt"

REM Copy plots
echo Copying plots...
copy /Y "Dataset\plots\confusion_matrix.png"    "Pipeline\plots\confusion_matrix.png"
copy /Y "Dataset\plots\roc_curve.png"           "Pipeline\plots\roc_curve.png"
copy /Y "Dataset\plots\training_curves.png"     "Pipeline\plots\training_curves.png"
copy /Y "Dataset\plots\sample_images.png"       "Pipeline\plots\sample_images.png"
copy /Y "Dataset\plots\prediction_samples.png"  "Pipeline\plots\prediction_samples.png"

echo.
echo Done! Now run these git commands:
echo.
echo   git add -A
echo   git commit -m "Add Render deployment config"
echo   git push
echo.
pause

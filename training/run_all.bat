@echo off
REM Runs the whole training pipeline (steps 1-8), stops at the first error.
REM Run step0_build_dataset.py ONCE before this.
REM Use from an activated venv:  C:\LivestockAI\.venv\Scripts\activate
cd /d %~dp0
python step1_check_dataset.py     || goto :fail
python step2_prepare_data.py      || goto :fail
python step3_train_ml.py          || goto :fail
python step4_train_dl.py          || goto :fail
python step6_export_hls.py        || goto :fail
python step7_train_mobilenetv2.py || goto :fail
python step8_fusion.py            || goto :fail
python step5_compare.py           || goto :fail
echo.
echo ===== ALL STEPS DONE =====
echo Chart: outputs\comparison.png    HLS files: hls_export\    PYNQ files: deploy\
goto :eof
:fail
echo.
echo ***** A STEP FAILED - read the error above *****
exit /b 1

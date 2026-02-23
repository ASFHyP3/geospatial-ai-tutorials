MODALITY=$1

cp generate_chips.py hwds
zip -r -9 hwds_$MODALITY.zip \
    hwds/$MODALITY/CHIPS_TM \
    hwds/$MODALITY/PLOTS \
    hwds/$MODALITY/SPLITS \
    hwds/$MODALITY/statistics.txt \
    hwds/SHP \
    hwds/generate_chips.py \
    -x "*.zip"

python train.py -s data/3dovs/sofa -m output/3dovs/sofa --start_checkpoint output/3dovs/sofa/chkpnt30000.pth --text Pikachu --include_mask -r 2 
python render_mask.py -m output/3dovs/sofa --text Pikachu --include_mask

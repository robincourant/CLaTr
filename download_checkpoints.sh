python -m gdown "https://drive.google.com/uc?id=1FqN-pa955Wvu3utGViUKiVfza6cL_W0D" # clatr-e100
python -m gdown "https://drive.google.com/uc?id=1YVrh7nhnujYMYbOQOUUek5ZTRn64K2gd" # clatr-text_encoder
python -m gdown "https://drive.google.com/uc?id=1LkwrknkQ7bURHl9Bqj2mDqEsGZJtCkpx" # clatr-traj_encoder
python -m gdown "https://drive.google.com/uc?id=1E-pui3CGMdW2Z7e85RYbHdMHl6W13-D0" # clatr-traj_decoder

mkdir checkpoints
mv clatr-e100.ckpt checkpoints
mv clatr_text_encoder.ckpt checkpoints
mv clatr_traj_encoder.ckpt checkpoints
mv clatr_traj_decoder.ckpt checkpoints

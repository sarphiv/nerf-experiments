# import pytorch_lightning as pl
from data_module import ImagePoseDataModule, QualitySchedule
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 5

quality_schedule = QualitySchedule(start_quality=0.1, max_quality_step=3)
dm = ImagePoseDataModule(
    image_width=800,
    image_height=800,
    scene_path="../data/lego",
    validation_fraction=0.05,
    validation_fraction_shuffle=1234,
    quality_schedule=quality_schedule,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
)

train_len = []

for epoch in tqdm(range(5)):
    train_dataloader = dm.train_dataloader()
    train_len.append(len(train_dataloader))

print(train_len)
plt.plot(train_len)
plt.show()

    
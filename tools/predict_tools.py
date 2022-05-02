import os
import datetime
import jittor as jt
from pathlib import Path
import zipfile

import cv2

from models.utils.utils import start_grad, stop_grad, weights_init_normal


@jt.single_process_scope()
def predict(generator, val_dataloader, args):

    stop_grad(generator)

    output_dir = Path(f"{args.output_path}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    f = zipfile.ZipFile(f'{args.output_path}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.zip', 'w', zipfile.ZIP_DEFLATED)

    # Iterate through val_dataloader
    for i, (_, real_A, photo_id) in enumerate(val_dataloader):
        fake_B = generator(real_A)

        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            filename = str(output_dir / f"{photo_id[idx]}.jpg")
            cv2.imwrite(filename,
                        fake_B[idx].transpose(1, 2, 0)[:, :, ::-1]) # BGR to RGB
            f.write(filename, arcname=f"{photo_id[idx]}.jpg")

    f.close()